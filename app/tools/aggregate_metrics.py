import os
import duckdb

def run(args: dict, dataset_path: str) -> list[dict]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        
    # Fetch column names from the CSV to match case-insensitively
    df_cols = duckdb.sql(f"SELECT * FROM read_csv_auto('{dataset_path}') LIMIT 0").columns
    
    group_col_raw = (
        args.get("group_by") or 
        args.get("group_cols") or 
        args.get("group_by_columns") or 
        args.get("group_by_column") or 
        args.get("groupby")
    )
    agg_col_raw = (
        args.get("metric") or 
        args.get("column") or 
        args.get("aggregate_column") or 
        args.get("metric_column") or
        "Revenue"
    )
    func_raw = args.get("function", "SUM")
    
    # Parse single or multiple functions (e.g. "MIN, MAX" -> ["MIN", "MAX"])
    if isinstance(func_raw, list):
        funcs = [f.strip().upper() for f in func_raw]
    else:
        funcs = [f.strip().upper() for f in str(func_raw).split(",") if f.strip()]
        
    if not funcs:
        funcs = ["SUM"]

    # Check if the query is a grand total / non-grouped aggregation
    is_grand_total = (
        group_col_raw is None or 
        str(group_col_raw).strip().lower() in ["none", "null", "", "false"]
    )
    
    # Match metric column case-insensitively against actual CSV headers
    agg_col = next((c for c in df_cols if c.lower() == str(agg_col_raw).lower()), agg_col_raw)
    
    # Build aggregation expressions (e.g. MIN("Revenue") as min_revenue, MAX("Revenue") as max_revenue)
    agg_exprs = [
        f'{f}("{agg_col}") as {f.lower()}_{agg_col.lower()}' if len(funcs) > 1 else f'{f}("{agg_col}") as result'
        for f in funcs
    ]
    select_aggs = ", ".join(agg_exprs)
    
    if is_grand_total:
        query = f"""
            SELECT {select_aggs} 
            FROM read_csv_auto('{dataset_path}')
        """
    else:
        group_cols = next((c for c in df_cols if c.lower() == str(group_col_raw).lower()), group_col_raw)
        query = f"""
            SELECT "{group_cols}", {select_aggs} 
            FROM read_csv_auto('{dataset_path}') 
            GROUP BY "{group_cols}"
        """
    
    result_df = duckdb.sql(query).df()
    return result_df.to_dict(orient="records")