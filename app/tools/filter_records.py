import duckdb
from typing import Dict, Any, List

def run(args: Dict[str, Any], dataset_path: str) -> List[Dict[str, Any]]:
    column = args.get("column") or args.get("col") or args.get("field")
    operator = args.get("operator") or args.get("op") or "="
    value = args.get("value") or args.get("val")
    
    if not column or value is None:
        raise ValueError(f"Missing filter parameters (column or value): {args}")
        
    op_map = {
        "equals": "=", "eq": "=", "=": "=",
        "greater_than": ">", "gt": ">", ">": ">",
        "less_than": "<", "lt": "<", "<": "<",
        "greater_than_equal": ">=", "gte": ">=", ">=": ">=",
        "less_than_equal": "<=", "lte": "<=", "<=": "<=",
        "not_equals": "!=", "neq": "!=", "!=": "!=",
        "contains": "LIKE"
    }
    sql_op = op_map.get(str(operator).lower(), "=")
    
    if sql_op == "LIKE":
        sql_val = f"'%{value}%'"
    elif isinstance(value, str):
        sql_val = f"'{value}'"
    else:
        sql_val = str(value)
        
    query = f"""
        SELECT * 
        FROM read_csv_auto('{dataset_path}') 
        WHERE {column} {sql_op} {sql_val}
    """
    return duckdb.sql(query).df().to_dict(orient="records")