from app.tools import aggregate_metrics, sort_and_rank, filter_records, group_by_analysis
import pandas as pd

def execute_row_count(args: dict, dataset_path: str) -> list:
    df = pd.read_csv(dataset_path)
    return [{"Total_Records": len(df)}]
TOOL_REGISTRY = {
    "aggregate_metrics": aggregate_metrics.run,
    "group_by_analysis": group_by_analysis.run,
    "sort_and_rank": sort_and_rank.run,
    "filter_records": filter_records.run,
    "get_row_count": execute_row_count,
}