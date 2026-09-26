# app/tools/group_by_analysis.py
from typing import List, Union, Dict, Any
import duckdb

def group_by_analysis(
    db_connection: duckdb.DuckDBPyConnection,
    group_columns: Union[List[str], str],
    metric_column: str,
    agg_func: str = "SUM"
) -> List[Dict[str, Any]]:
    """Groups dataset records by one or multiple columns and calculates aggregate metrics."""
    if isinstance(group_columns, str):
        group_columns = [col.strip() for col in group_columns.split(",") if col.strip()]

    if not group_columns:
        raise ValueError("At least one valid group column must be provided.")

    quoted_cols = [f'"{col}"' for col in group_columns]
    group_by_clause = ", ".join(quoted_cols)
    quoted_metric = f'"{metric_column.strip()}"'
    clean_agg = agg_func.strip().upper()

    query = f"""
        SELECT 
            {group_by_clause}, 
            {clean_agg}({quoted_metric}) AS total_{metric_column.strip().lower().replace(' ', '_')}
        FROM dataset
        GROUP BY {group_by_clause}
        ORDER BY 2 DESC
    """

    df = db_connection.execute(query).fetchdf()
    return df.to_dict(orient="records")

def run(args: dict, dataset_path: str) -> List[Dict[str, Any]]:
    """Standardized entry point for TOOL_REGISTRY."""
    conn = duckdb.connect(database=":memory:")
    conn.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{dataset_path}')")
    
    group_cols = args.get("group_by_columns") or args.get("group_columns")
    metric_col = args.get("metric_column") or args.get("aggregate_column")
    agg_f = args.get("agg_func") or args.get("aggregation") or args.get("aggregation_function", "SUM")

    return group_by_analysis(
        db_connection=conn,
        group_columns=group_cols,
        metric_column=metric_col,
        agg_func=agg_f
    )