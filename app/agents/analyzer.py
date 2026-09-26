import os
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent

try:
    from pydantic_ai.models.gemini import GeminiModel
except ImportError:
    from pydantic_ai.models.google import GoogleModel as GeminiModel

# -----------------------------------------------------------------------------
# Tool Taxonomy & Structured Schemas
# -----------------------------------------------------------------------------
class ToolName(str, Enum):
    GET_ROW_COUNT = "get_row_count"
    AGGREGATE_METRICS = "aggregate_metrics"
    GROUP_BY_ANALYSIS = "group_by_analysis"
    SORT_AND_RANK = "sort_and_rank"
    FILTER_RECORDS = "filter_records"

class AnalyticalPlan(BaseModel):
    tool_name: ToolName = Field(
        description="The target analytical tool to execute against DuckDB."
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="""Arguments required for tool execution:
        - get_row_count: {}
        - aggregate_metrics: {'aggregate_column': str, 'aggregation_function': 'SUM'|'AVG'|'MIN'|'MAX'|'COUNT'}
        - group_by_analysis: {'group_by_columns': List[str], 'metric_column': str, 'agg_func': 'SUM'|'AVG'|'MIN'|'MAX'|'COUNT'}
        - sort_and_rank: {'entity_column': str, 'metric_column': str, 'limit': int, 'order': 'DESC'|'ASC'}
        - filter_records: {'filter_column': str, 'operator': '=', 'value': str}
        """
    )

# -----------------------------------------------------------------------------
# Heuristic Rule Engine (Fallback Router)
# -----------------------------------------------------------------------------
def heuristic_fallback(query: str, schema_columns: List[str]) -> AnalyticalPlan:
    """Provides rule-based query parsing when LLM calls hit rate limits or fail."""
    q_lower = query.lower()
    
    # Non-ID columns
    non_id_cols = [c for c in schema_columns if not (c.lower().endswith("_id") or c.lower() == "id")]
    if not non_id_cols:
        non_id_cols = schema_columns

    # Identify primary numeric metric column
    metric_col = next(
        (c for c in schema_columns if any(m in c.lower() for m in ["revenue", "sales", "amount", "total", "price", "units"])),
        schema_columns[-1]
    )

    # Row count queries
    if any(k in q_lower for k in ["how many records", "total rows", "row count", "total records", "number of rows"]):
        return AnalyticalPlan(tool_name=ToolName.GET_ROW_COUNT, tool_args={})

    # PRIORITY MULTI-COLUMN CHECK: Extract all schema columns mentioned in the query
    matched_cols = []
    for col in schema_columns:
        col_clean = col.replace("_", " ").lower()
        if col_clean in q_lower or col.lower() in q_lower:
            matched_cols.append(col)

    # Separate metric from categorical dimensions
    group_candidates = [c for c in matched_cols if c != metric_col and not (c.lower().endswith("_id") or c.lower() == "id")]

    # If any categorical columns matched or 'by'/'breakdown' was requested, route to group_by_analysis
    if group_candidates or "by" in q_lower or "breakdown" in q_lower:
        return AnalyticalPlan(
            tool_name=ToolName.GROUP_BY_ANALYSIS,
            tool_args={
                "group_by_columns": group_candidates if group_candidates else [non_id_cols[0]],
                "metric_column": metric_col,
                "agg_func": "SUM"
            }
        )

    return AnalyticalPlan(
        tool_name=ToolName.GROUP_BY_ANALYSIS,
        tool_args={
            "group_by_columns": [non_id_cols[0]],
            "metric_column": metric_col,
            "agg_func": "SUM"
        }
    )

# -----------------------------------------------------------------------------
# Agent Definitions
# -----------------------------------------------------------------------------
raw_model_env = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
for prefix in ["google-gla:", "google-vertex:", "google-cloud:"]:
    if raw_model_env.startswith(prefix):
        raw_model_env = raw_model_env[len(prefix):]

model = GeminiModel(raw_model_env)

analytics_agent = Agent(
    model,
    system_prompt=(
        "You are an expert Data Analytics Router. Analyze the user query and dataset schema columns, "
        "then select the best tool and arguments to answer the prompt.\n\n"
        "STRICT ROUTING RULES:\n"
        "1. GROUP BY / BREAKDOWNS:\n"
        "   - Use `group_by_analysis` whenever a query asks to break down, group, or summarize metrics by ONE or MULTIPLE columns (e.g., 'revenue by Region and Marketing Channel').\n"
        "   - Extract ALL requested categorical dimensions into `group_by_columns` as a list of exact column names (e.g., ['Region', 'Marketing Channel']).\n"
        "   - NEVER pick Primary Keys / Unique IDs (e.g., 'Transaction_ID') as grouping columns.\n\n"
        "2. METRIC MATCHING:\n"
        "   - Match requested metrics (e.g., 'revenue', 'sales') to exact numeric schema column names.\n"
    )
)

synthesizer_agent = Agent(
    model,
    system_prompt=(
        "You are a Senior Business Intelligence Lead. Given execution results from a data query, "
        "provide a concise executive summary highlighting key figures and top dimensions."
    )
)