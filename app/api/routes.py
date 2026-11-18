import os
import json
import time
import tempfile
from collections import deque
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


def find_join_path(tables_meta: dict, start_table: str, target_table: str) -> list[tuple[str, str, str]]:
    """
    Dataset-agnostic BFS shortest-path finder between two tables using shared *_id keys.
    Returns a list of (left_table, right_table, join_key) tuples.
    """
    if start_table == target_table:
        return []

    queue = deque([(start_table, [])])
    visited = {start_table}

    while queue:
        curr_table, path = queue.popleft()
        curr_cols = {
            c.lower(): c
            for c in tables_meta[curr_table]["columns"]
            if c.lower().endswith("_id") or c.lower() == "id"
        }

        for next_table, meta in tables_meta.items():
            if next_table in visited:
                continue
            next_cols = {
                c.lower(): c
                for c in meta["columns"]
                if c.lower().endswith("_id") or c.lower() == "id"
            }

            # Find shared ID columns between curr_table and next_table
            shared_keys = set(curr_cols.keys()) & set(next_cols.keys())
            if shared_keys:
                # Prefer specific foreign keys over generic 'id' or 'user_id'
                preferred_key = sorted(
                    shared_keys,
                    key=lambda k: (k == "id", k == "user_id", len(k))
                )[0]
                new_path = path + [(curr_table, next_table, curr_cols[preferred_key])]
                if next_table == target_table:
                    return new_path
                visited.add(next_table)
                queue.append((next_table, new_path))

    return []


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

    # Build combined schema context for multi-table or single-table routing
    tables_meta = schema_summary.get("tables", {})
    if len(tables_meta) > 1:
        schema_context_str = {tname: meta["columns"] for tname, meta in tables_meta.items()}
        all_columns_flat = [col for meta in tables_meta.values() for col in meta["columns"]]
    else:
        schema_context_str = schema_summary["columns"]
        all_columns_flat = schema_summary["columns"]

    # 1. Routing Agent Execution
    try:
        plan = await analytics_agent.run(
            user_prompt=f"User Query: {payload.query}\nSchema Context: {schema_context_str}"
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
        fallback_plan = heuristic_fallback(payload.query, all_columns_flat)
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
        elif tool_name in ["aggregate_metric", "aggregate_metrics"]:
            metric_col = tool_args.get("metric_column") or tool_args.get("aggregate_column") or schema_summary['columns'][0]
            agg_func = str(tool_args.get("aggregation") or tool_args.get("aggregation_function", "SUM")).upper()
            if agg_func == "COUNT_DISTINCT":
                sql_query = f'SELECT COUNT(DISTINCT "{metric_col}") AS distinct_{metric_col} FROM active_dataset'
            else:
                sql_query = f'SELECT ROUND({agg_func}("{metric_col}"), 2) AS result_{metric_col} FROM active_dataset'
            raw_res = conn.execute(sql_query).df()

        # --- TOOL 3: Multi-Dimensional Group By (Dataset-Agnostic Multi-Table Support) ---
        elif tool_name == "group_by_analysis":
            raw_dims = tool_args.get("group_by_columns", [])
            raw_metric = tool_args.get("metric_column")
            agg_func = str(tool_args.get("aggregation") or tool_args.get("agg_func", "SUM")).upper()

            if len(tables_meta) > 1:
                query_lower = payload.query.lower()

                # 1. Identify requested metric & pick the right Fact table dynamically
                metric_keywords = ["line_item_total", "revenue", "sales", "amount", "total", "price", "profit", "quantity", "units"]
                if raw_metric and str(raw_metric).lower() in query_lower:
                    metric_keywords.insert(0, str(raw_metric).lower())

                best_fact = None
                best_metric_col = None
                best_score = -1

                for tname, tmeta in tables_meta.items():
                    score = 0
                    matched_m = None
                    if "fact" in tname.lower():
                        score += 2
                    for col, dtype in tmeta["schema_types"].items():
                        col_l = col.lower()
                        is_numeric = any(nt in dtype.upper() for nt in ["INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT"])
                        if is_numeric and not col_l.endswith("_id") and col_l != "id":
                            if any(mk in col_l for mk in metric_keywords):
                                score += 5
                                if not matched_m:
                                    matched_m = col
                            elif not matched_m:
                                matched_m = col
                    # Boost fact table if query mentions words related to the table domain
                    if any(w in query_lower for w in ["sale", "sales", "transaction", "purchase", "product", "listing", "order"]) and "transaction" in tname.lower():
                        score += 10
                    if any(w in query_lower for w in ["job", "application", "applicant", "hire", "hiring", "salary"]) and "application" in tname.lower():
                        score += 10

                    if score > best_score:
                        best_score = score
                        best_fact = tname
                        best_metric_col = matched_m

                fact_table = best_fact or list(tables_meta.keys())[0]

                # 2. Check if a time-based grouping (monthly, yearly, daily, weekly) is requested
                time_trunc_unit = None
                time_label = None
                time_fmt = "%Y-%m"
                if "monthly" in query_lower or "by month" in query_lower or "per month" in query_lower:
                    time_trunc_unit, time_label, time_fmt = "month", "sales_month", "%Y-%m"
                elif "yearly" in query_lower or "annual" in query_lower or "by year" in query_lower:
                    time_trunc_unit, time_label, time_fmt = "year", "sales_year", "%Y"
                elif "weekly" in query_lower or "by week" in query_lower:
                    time_trunc_unit, time_label, time_fmt = "week", "sales_week", "%Y-%m-%d"
                elif "daily" in query_lower or "by day" in query_lower:
                    time_trunc_unit, time_label, time_fmt = "day", "sales_day", "%Y-%m-%d"

                dim_selects = []
                dim_group_by = []
                where_conditions = []
                joined_tables = {fact_table}
                join_clauses = []

                if time_trunc_unit:
                    # 1. Prioritize columns with actual DATE/TIMESTAMP DuckDB types
                    date_cols = [
                        c for c, t in tables_meta[fact_table]["schema_types"].items()
                        if any(dt in t.upper() for dt in ["DATE", "TIME"])
                    ]
                    # 2. Fallback to columns explicitly named *_date, *_at, or timestamp
                    if not date_cols:
                        date_cols = [
                            c for c in tables_meta[fact_table]["columns"]
                            if c.lower().endswith("_date") or c.lower().endswith("_at") or "date" in c.lower() or "timestamp" in c.lower()
                        ]
                    if date_cols:
                        dcol = date_cols[0]
                        time_expr = f"STRFTIME(DATE_TRUNC('{time_trunc_unit}', TRY_CAST({fact_table}.\"{dcol}\" AS TIMESTAMP)), '{time_fmt}')"
                        dim_selects.append(f"{time_expr} AS {time_label}")
                        dim_group_by.append(time_expr)
                        where_conditions.append(f'TRY_CAST({fact_table}."{dcol}" AS TIMESTAMP) IS NOT NULL')

                # 3. Dynamically resolve target dimension tables & columns from query + raw_dims
                search_terms = [query_lower]
                if isinstance(raw_dims, list):
                    search_terms.extend([str(d).lower().replace("_", " ") for d in raw_dims])
                elif isinstance(raw_dims, str):
                    search_terms.append(raw_dims.lower().replace("_", " "))

                resolved_dimensions = []
                for tname, tmeta in tables_meta.items():
                    if tname == fact_table:
                        continue
                    # Match table name tokens (e.g., "listing" matches "dim_listing")
                    clean_tname = tname.lower().replace("dim_", "").replace("fact_", "").replace("_", " ")
                    table_mentioned = clean_tname in query_lower or any(clean_tname in st for st in search_terms)

                    text_cols = [
                        c for c, dtype in tmeta["schema_types"].items()
                        if ("VARCHAR" in dtype.upper() or "TEXT" in dtype.upper())
                        and not c.lower().endswith("_id")
                        and not any(skip in c.lower() for skip in ["email", "website", "url", "date", "_at"])
                    ]

                    for col in text_cols:
                        col_clean = col.lower().replace("_", " ")
                        if col_clean in query_lower or any(col_clean == st for st in search_terms):
                            if (tname, col) not in resolved_dimensions:
                                resolved_dimensions.append((tname, col))

                    # If the dimension table was explicitly named (e.g. "product listing") but no specific column matched, pick its primary descriptor (e.g. 'title' or 'name')
                    if table_mentioned and not any(dt == tname for dt, _ in resolved_dimensions) and text_cols:
                        primary_desc = next((c for c in text_cols if any(k in c.lower() for k in ["title", "name", "category", "type"])), text_cols[0])
                        resolved_dimensions.append((tname, primary_desc))

                # Fallback if still no dimensions found: pick the first reachable dimension table
                if not resolved_dimensions:
                    for tname, tmeta in tables_meta.items():
                        if tname != fact_table and find_join_path(tables_meta, fact_table, tname):
                            text_cols = [c for c, dtype in tmeta["schema_types"].items() if "VARCHAR" in dtype.upper() and not c.lower().endswith("_id")]
                            if text_cols:
                                resolved_dimensions.append((tname, text_cols[0]))
                                break

                # 4. Build BFS multi-hop JOIN clauses for every resolved dimension table
                for dtable, dcol in resolved_dimensions:
                    path = find_join_path(tables_meta, fact_table, dtable)
                    if not path and dtable != fact_table:
                        continue
                    for left_t, right_t, join_key in path:
                        if right_t not in joined_tables:
                            joined_tables.add(right_t)
                            join_clauses.append(f'JOIN {right_t} ON {left_t}."{join_key}" = {right_t}."{join_key}"')
                    dim_selects.append(f'{dtable}."{dcol}"')
                    dim_group_by.append(f'{dtable}."{dcol}"')
                    where_conditions.append(f'{dtable}."{dcol}" IS NOT NULL')

                # 5. Build Metric SELECT clause
                if best_metric_col:
                    metric_alias = f"total_{best_metric_col.lower()}"
                    metrics_sql = f'ROUND({agg_func}({fact_table}."{best_metric_col}"), 2) AS {metric_alias}'
                else:
                    metric_alias = "total_records"
                    metrics_sql = "COUNT(*) AS total_records"

                where_sql = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
                order_sql = f"{dim_group_by[0]} ASC, {metric_alias} DESC" if time_trunc_unit else f"{metric_alias} DESC"

                sql_query = f"""
                    SELECT {", ".join(dim_selects)}, {metrics_sql}
                    FROM {fact_table}
                    {" ".join(join_clauses)}
                    {where_sql}
                    GROUP BY {", ".join(dim_group_by)}
                    ORDER BY {order_sql}
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
        elif tool_name in ["top_n_ranking", "sort_and_rank"]:
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