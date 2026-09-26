from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class LoadAndProfileArgs(BaseModel):
    """Schema for initial data ingestion and profiling."""
    file_path: str = Field(..., description="Path to the CSV or Excel file.")

class CleanMissingValuesArgs(BaseModel):
    """Schema for handling missing data."""
    strategy: Literal["drop", "mean", "median", "forward_fill"] = Field(..., description="Imputation strategy to apply.")
    columns: Optional[List[str]] = Field(None, description="Specific columns to clean. If none, apply to all applicable columns.")

class AggregateMetricsArgs(BaseModel):
    """Schema for single-metric aggregations."""
    aggregate_column: str = Field(..., description="The numeric column to aggregate.")
    aggregation_function: Literal["SUM", "AVG", "MIN", "MAX", "COUNT"] = Field(..., description="SQL aggregation function.")

class GroupByAnalysisArgs(BaseModel):
    """Schema for OLAP group-by and breakdown operations across one or more dimensions."""
    group_columns: List[str] = Field(..., description="List of categorical columns to group by (e.g., ['Region', 'Marketing Channel']).")
    metric_column: str = Field(..., description="The numeric column to aggregate.")
    agg_func: Literal["SUM", "AVG", "MIN", "MAX", "COUNT"] = Field("SUM", description="Aggregation function.")

class FilterRecordsArgs(BaseModel):
    """Schema for applying conditional WHERE filters."""
    filter_column: str = Field(..., description="Column to apply the filter on.")
    operator: Literal["=", ">", "<", ">=", "<=", "!=", "LIKE"] = Field(..., description="Comparison operator.")
    value: str = Field(..., description="Value to compare against. Use ISO format for dates.")