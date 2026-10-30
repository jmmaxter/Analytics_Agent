import os
import json
import time
import tempfile
import duckdb
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.api import DatasetSummary, QueryRequest, QueryResponse
from app.services.session_manager import session_manager
from app.agents.analyzer import analytics_agent, synthesizer_agent, AnalyticalPlan, heuristic_fallback

router = APIRouter(prefix="/api/v1", tags=["Analytics Workspace"])


@router.get("/health", status_code=200)
async def health_check():
    """Health check endpoint for Render service uptime monitoring."""
    return {"status": "healthy", "timestamp": time.time()}


def resolve_columns(requested_cols: list[str] | str, available_cols: list[str]) -> list[str]:
    """Resolves requested column names against schema, preventing accidental ID column grouping."""
    if isinstance(requested_cols, str):
        requested_cols = [requested_cols]
        
    resolved = []
    for req in requested_cols:
        req_clean = req.lower().strip()
        # 1. Direct or case-insensitive match
        match = next((c for c in available_cols if c.lower() == req_clean), None)
        # 2. Substring match
        if not match:
            match = next((c for c in available_cols if req_clean in c.lower()), None)
            
        if match:
            resolved.append(match)
            
    # Filter out primary keys if multiple columns exist
    filtered = [c for c in resolved if not (c.lower().endswith("_id") or c.lower() == "id")]
    return filtered if filtered else (resolved if resolved else [available_cols[0]])


@router.post("/dataset/upload", response_model=DatasetSummary, status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a CSV or Excel file, initializes an isolated DuckDB workspace, and returns schema metadata."""
    allowed_extensions = [".csv", ".xlsx", ".xls"]
    filename = file.filename or ""
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type '{file_ext}'. Allowed types: {', '.join(allowed_extensions)}"
        )

    tmp_file_path = None
    try:
        # Write temporary file and explicitly close handle before session creation
        # to prevent file lock contention in Linux/Render environments
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        summary = session_manager.create_session(tmp_file_path, filename)
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@router.post("/query", response_model=QueryResponse)
async def execute_query(payload: QueryRequest):
    """Processes a natural language query against the dataset tied to the active session."""
    session = session_manager.get_session(payload.session_id)
    if not session:
        raise HTTPException(
            status_code=404, 
            detail="Session expired or not found. Please re-upload your dataset."
        )

    start_time = time.time()
    db_path = session["db_path"]
    schema_summary = session["summary"]

    # 1. Routing Agent Execution
    try:
        plan = await analytics_agent.run(
            user_prompt=f"User Query: {payload.query}\nSchema Context: {schema_summary['columns']}"
        )
        plan_data = getattr(plan, "output", getattr(plan, "data", plan))

        # Parse string output into AnalyticalPlan if returned as raw JSON text
        if isinstance(plan_data, str):
            clean_json = plan_data.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = AnalyticalPlan.model_validate_json(clean_json)
            tool_name = parsed.tool_name.value if hasattr(parsed.tool_name, "value") else str(parsed.tool_name)
            tool_args = parsed.tool_args
        else:
            tool_name = str(getattr(plan_data, "tool_name", "unknown")).lower()
            if hasattr(tool_name, "value"):
                tool_name = tool_name.value
            tool_args = getattr(plan_data, "tool_args", {}) or {}

    except Exception:
        # Fallback to rule engine if LLM call or JSON parsing fails
        fallback_plan = heuristic_fallback(payload.query, schema_summary['columns'])
        tool_name = fallback_plan.tool_name.value if hasattr(fallback_plan.tool_name, "value") else str(fallback_plan.tool_name)
        tool_args = fallback_plan.tool_args

    # 2. Dynamic DuckDB SQL Generator & Executor
    conn = None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # --- TOOL 1: Row Count ---
        if tool_name == "get_row_count":
            raw_res = conn.execute("SELECT COUNT(*) as total_records FROM active_dataset").df()

        # --- TOOL 2: Single Metric Aggregation ---
        elif tool_name == "aggregate_metric":
            metric_col = tool_args.get("metric_column", schema_summary['columns'][0])
            agg_func = tool_args.get("aggregation", "SUM").upper()
            if agg_func == "COUNT_DISTINCT":
                sql_query = f'SELECT COUNT(DISTINCT "{metric_col}") AS distinct_{metric_col} FROM active_dataset'
            else:
                sql_query = f'SELECT ROUND({agg_func}("{metric_col}"), 2) AS result_{metric_col} FROM active_dataset'
            raw_res = conn.execute(sql_query).df()

        # --- TOOL 3: Multi-Dimensional Group By (Multi-Table Star Schema Support) ---
        elif tool_name == "group_by_analysis":
            raw_dims = tool_args.get("group_by_columns", [])
            raw_metric = tool_args.get("metric_column")
            agg_func = tool_args.get("aggregation", "SUM").upper()

            tables_meta = schema_summary.get("tables", {})
            
            if len(tables_meta) > 1:
                fact_table = next((t for t in tables_meta.keys() if "fact" in t), list(tables_meta.keys())[0])
                dim_tables = [t for t in tables_meta.keys() if t != fact_table]
                
                query_lower = payload.query.lower()
                resolved_dimensions = []
                joined_tables = set()
                join_clauses = []

                # Map business terms to their respective dimension tables, columns, and foreign keys
                dim_mapping = {
                    "product_category": ("dim_products", "Product_Category", "Product_ID"),
                    "category": ("dim_products", "Product_Category", "Product_ID"),
                    "region": ("dim_stores", "Region", "Store_ID"),
                    "store_location": ("dim_stores", "Store_Location", "Store_ID"),
                    "customer_segment": ("dim_customers", "Customer_Segment", "Customer_ID"),
                    "sales_rep": ("dim_sales_rep", "Sales_Rep", "Rep_ID"),
                    "marketing_channel": ("dim_sales_rep", "Marketing_Channel", "Rep_ID"),
                }

                # Check prompt or raw_dims for matching dimensions
                search_targets = [query_lower]
                if isinstance(raw_dims, list):
                    search_targets.extend([str(d).lower() for d in raw_dims])
                elif isinstance(raw_dims, str):
                    search_targets.append(raw_dims.lower())

                for term, (dtable, dcol, fkey) in dim_mapping.items():
                    if any(term in target for target in search_targets):
                        if dtable in tables_meta and (dtable, dcol) not in resolved_dimensions:
                            resolved_dimensions.append((dtable, dcol))
                            if dtable not in joined_tables and dtable != fact_table:
                                joined_tables.add(dtable)
                                join_clauses.append(f'JOIN {dtable} ON {fact_table}."{fkey}" = {dtable}."{fkey}"')

                # Fallback if no specific dimension was matched
                if not resolved_dimensions:
                    dtable = dim_tables[0]
                    text_cols = [c for c, t in tables_meta[dtable]["schema_types"].items() if "VARCHAR" in t.upper() or "TEXT" in t.upper()]
                    dcol = text_cols[0] if text_cols else tables_meta[dtable]["columns"][1]
                    fkey = next((c for c in tables_meta[fact_table]["columns"] if c in tables_meta[dtable]["columns"] or c.endswith("_id")), f"{dtable.replace('dim_', '')}_id")
                    resolved_dimensions.append((dtable, dcol))
                    join_clauses.append(f'JOIN {dtable} ON {fact_table}."{fkey}" = {dtable}."{fkey}"')

                dim_selects = [f'{dt}."{col}"' for dt, col in resolved_dimensions]
                dim_group_by = [f'{dt}."{col}"' for dt, col in resolved_dimensions]
                where_not_null = " AND ".join([f'{dt}."{col}" IS NOT NULL' for dt, col in resolved_dimensions])

                # Resolve metrics (supporting single or multi-metric queries like revenue and profit)
                fact_cols = tables_meta[fact_table]["columns"]
                select_metrics_clause = []
                
                if "profit" in query_lower:
                    select_metrics_clause.append(f'ROUND({agg_func}({fact_table}."Profit"), 2) AS total_profit')
                if "revenue" in query_lower or "sales" in query_lower:
                    select_metrics_clause.append(f'ROUND({agg_func}({fact_table}."Revenue"), 2) AS total_revenue')
                if "unit" in query_lower:
                    select_metrics_clause.append(f'ROUND({agg_func}({fact_table}."Units_Sold"), 2) AS total_units_sold')

                if not select_metrics_clause:
                    metric_col = next((c for c in fact_cols if any(m in c.lower() for m in ["revenue", "profit", "sales", "amount"])), fact_cols[-1])
                    select_metrics_clause.append(f'ROUND({agg_func}({fact_table}."{metric_col}"), 2) AS total_{metric_col.lower()}')

                metrics_sql = ", ".join(select_metrics_clause)
                primary_metric_alias = select_metrics_clause[0].split(" AS ")[1]

                sql_query = f"""
                    SELECT {", ".join(dim_selects)}, {metrics_sql}
                    FROM {fact_table}
                    {" ".join(join_clauses)}
                    WHERE {where_not_null}
                    GROUP BY {", ".join(dim_group_by)}
                    ORDER BY {primary_metric_alias} DESC
                """
            else:
                # SINGLE DATASET MODE
                non_id_cols = [c for c in schema_summary['columns'] if not (c.lower().endswith("_id") or c.lower() == "id")]
                if not non_id_cols:
                    non_id_cols = schema_summary['columns']

                dimensions = resolve_columns(raw_dims, schema_summary['columns'])
                dimensions = [d for d in dimensions if not (d.lower().endswith("_id") or d.lower() == "id")]
                
                if not dimensions:
                    prompt_lower = payload.query.lower()
                    matched = [c for c in non_id_cols if c.lower() in prompt_lower]
                    dimensions = matched if matched else [non_id_cols[0]]

                metric_col = None
                if raw_metric:
                    resolved_m = resolve_columns([raw_metric], schema_summary['columns'])
                    if resolved_m:
                        metric_col = resolved_m[0]
                
                if not metric_col:
                    metric_col = next(
                        (c for c in schema_summary['columns'] if any(m in c.lower() for m in ["revenue", "sales", "amount", "total", "price", "profit"])),
                        None
                    )

                escaped_dims = [f'"{d}"' for d in dimensions]
                dim_clause = ", ".join(escaped_dims)
                where_clause = " AND ".join([f"{d} IS NOT NULL" for d in escaped_dims])

                if metric_col and metric_col not in dimensions:
                    sql_query = f"""
                        SELECT {dim_clause}, ROUND({agg_func}("{metric_col}"), 2) AS total_{metric_col}
                        FROM active_dataset
                        WHERE {where_clause}
                        GROUP BY {dim_clause}
                        ORDER BY total_{metric_col} DESC
                    """
                else:
                    sql_query = f"""
                        SELECT {dim_clause}, COUNT(*) AS record_count
                        FROM active_dataset
                        WHERE {where_clause}
                        GROUP BY {dim_clause}
                        ORDER BY record_count DESC
                    """
            
            raw_res = conn.execute(sql_query).df()

        # --- TOOL 4: Top N Leaderboard Ranking ---
        elif tool_name == "top_n_ranking":
            entity_col = tool_args.get("entity_column", schema_summary['columns'][0])
            metric_col = tool_args.get("metric_column", schema_summary['columns'][-1])
            n_limit = tool_args.get("limit", 5)
            order = str(tool_args.get("order", "DESC")).upper()
            
            sql_query = f"""
                SELECT "{entity_col}", ROUND(SUM("{metric_col}"), 2) AS total_{metric_col}
                FROM active_dataset
                WHERE "{entity_col}" IS NOT NULL
                GROUP BY "{entity_col}"
                ORDER BY total_{metric_col} {order}
                LIMIT {n_limit}
            """
            raw_res = conn.execute(sql_query).df()

        # --- TOOL 5: Period-over-Period (MoM/YoY Growth) ---
        elif tool_name == "period_over_period":
            date_col = tool_args.get("date_column")
            metric_col = tool_args.get("metric_column")
            time_frame = str(tool_args.get("time_frame", "MoM")).upper()

            if not date_col:
                date_cols = [c for c, t in schema_summary["schema_types"].items() if "DATE" in t.upper() or "TIME" in t.upper()]
                date_col = date_cols[0] if date_cols else schema_summary['columns'][0]
            if not metric_col:
                metric_col = schema_summary['columns'][-1]

            trunc_unit = "year" if time_frame == "YOY" else "month"
            date_format = '%Y' if time_frame == "YOY" else '%Y-%m'
            
            sql_query = f"""
            WITH aggregated_periods AS (
                SELECT 
                    DATE_TRUNC('{trunc_unit}', TRY_CAST("{date_col}" AS DATE)) AS period_date,
                    ROUND(SUM("{metric_col}"), 2) AS current_amount
                FROM active_dataset
                WHERE TRY_CAST("{date_col}" AS DATE) IS NOT NULL
                GROUP BY 1
            )
            SELECT 
                STRFTIME(period_date, '{date_format}') AS period,
                current_amount,
                LAG(current_amount) OVER (ORDER BY period_date) AS previous_period_amount,
                ROUND(current_amount - LAG(current_amount) OVER (ORDER BY period_date), 2) AS net_variance,
                ROUND(((current_amount - LAG(current_amount) OVER (ORDER BY period_date)) / NULLIF(LAG(current_amount) OVER (ORDER BY period_date), 0)) * 100, 2) AS growth_percentage
            FROM aggregated_periods
            ORDER BY period_date DESC
            """
            raw_res = conn.execute(sql_query).df()

        # --- TOOL 6: Cross-Entity Performance ---
        elif tool_name == "cross_entity_performance":
            entity_col = tool_args.get("entity_column", schema_summary['columns'][0])
            metric_col = tool_args.get("metric_column", schema_summary['columns'][-1])
            
            sql_query = f"""
            SELECT 
                "{entity_col}" AS entity_name,
                COUNT(*) AS total_transactions,
                ROUND(SUM("{metric_col}"), 2) AS total_performance,
                ROUND(AVG("{metric_col}"), 2) AS avg_transaction_value,
                ROUND(SUM("{metric_col}") * 100.0 / NULLIF(SUM(SUM("{metric_col}")) OVER(), 0), 2) AS market_share_percentage
            FROM active_dataset
            WHERE "{entity_col}" IS NOT NULL
            GROUP BY "{entity_col}"
            ORDER BY total_performance DESC
            """
            raw_res = conn.execute(sql_query).df()

        # --- General Dynamic Fallback ---
        else:
            raw_res = conn.execute("SELECT * FROM active_dataset").df()
            
        raw_data = raw_res.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database execution error: {str(e)}")
    finally:
        if conn:
            conn.close()

    # 3. Insight Synthesis Execution
    try:
        insight_response = await synthesizer_agent.run(
            user_prompt=f"User Query: {payload.query}\nTool Output: {raw_data}"
        )
        insight_text = getattr(insight_response, "output", getattr(insight_response, "data", str(insight_response)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight synthesis failed: {str(e)}")

    execution_time = round((time.time() - start_time) * 1000, 2)

    return QueryResponse(
        status="success",
        session_id=payload.session_id,
        original_query=payload.query,
        selected_tool=tool_name,
        tool_args=tool_args,
        insight=str(insight_text),
        raw_data=raw_data,
        execution_time_ms=execution_time
    )


@router.delete("/dataset/{session_id}", status_code=200)
async def clear_session(session_id: str):
    """Purges active session workspace and deletes database file."""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session cleared successfully."}

@router.post("/dataset/upload-multi", response_model=DatasetSummary, status_code=201)
async def upload_multi_dataset(files: list[UploadFile] = File(...)):
    """Uploads multiple CSV/Excel files for multi-table Star Schema BI projects."""
    allowed_extensions = [".csv", ".xlsx", ".xls"]
    tmp_file_paths = []
    file_data_list = []

    try:
        for file in files:
            filename = file.filename or ""
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"Invalid file type '{file_ext}'.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_paths.append(tmp_file.name)
                file_data_list.append((tmp_file.name, filename))

        summary = session_manager.create_multi_table_session(file_data_list)
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process datasets: {str(e)}")
    finally:
        for path in tmp_file_paths:
            if os.path.exists(path):
                os.remove(path)