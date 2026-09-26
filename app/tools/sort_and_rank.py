import duckdb
from typing import Dict, Any, List

def run(args: Dict[str, Any], dataset_path: str) -> List[Dict[str, Any]]:
    group_keys = (
        args.get("group_by_columns") or 
        args.get("group_cols") or 
        args.get("group_by") or 
        args.get("group_by_column") or 
        args.get("groupby_column") or 
        args.get("groupby") or 
        args.get("columns")
    )
    if not group_keys:
        raise ValueError(f"Missing grouping columns for sorting: {args}")
        
    group_cols = ", ".join(group_keys) if isinstance(group_keys, list) else str(group_keys)
    
    agg_col = (
        args.get("aggregate_column") or 
        args.get("column") or 
        args.get("metric") or 
        args.get("metric_column")
    )
    if not agg_col:
        raise ValueError(f"Missing aggregate column for sorting: {args}")
        
    func = str(
        args.get("aggregation_function") or 
        args.get("function") or 
        args.get("operation") or 
        "SUM"
    ).upper()
    
    limit = int(args.get("limit") or args.get("top_n") or 1)
    sort_order = "ASC" if bool(args.get("ascending", False)) else "DESC"
    
    query = f"""
        SELECT {group_cols}, {func}({agg_col}) as result 
        FROM read_csv_auto('{dataset_path}') 
        GROUP BY {group_cols}
        ORDER BY result {sort_order}
        LIMIT {limit}
    """
    return duckdb.sql(query).df().to_dict(orient="records")