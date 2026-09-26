# app/tools/registry.py
import pandas as pd
from app.tools.aggregate_metrics import run as aggregate_metrics_run
from app.tools.sort_and_rank import run as sort_and_rank_run
from app.tools.filter_records import run as filter_records_run
from app.tools.group_by_analysis import run as group_by_analysis_run

def execute_row_count(args: dict, dataset_path: str) -> list:
    df = pd.read_csv(dataset_path)
    return [{"Total_Records": len(df)}]

TOOL_REGISTRY = {
    "aggregate_metrics": aggregate_metrics_run,
    "group_by_analysis": group_by_analysis_run,
    "sort_and_rank": sort_and_rank_run,
    "filter_records": filter_records_run,
    "get_row_count": execute_row_count,
}