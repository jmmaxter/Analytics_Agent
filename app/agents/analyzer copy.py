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
    AGGREGATE_METRIC = "aggregate_metric"
    GROUP_BY_ANALYSIS = "group_by_analysis"
    TOP_N_RANKING = "top_n_ranking"
    PERIOD_OVER_PERIOD = "period_over_period"
    CROSS_ENTITY_PERFORMANCE = "cross_entity_performance"

class AnalyticalPlan(BaseModel):
    tool_name: ToolName = Field(
        description="The target analytical tool to execute against DuckDB."
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="""Arguments required for tool execution:
        - get_row_count: {}
        - aggregate_metric: {'metric_column': str, 'aggregation': 'SUM'|'AVG'|'MIN'|'MAX'|'COUNT_DISTINCT'}
        - group_by_analysis: {'group_by_columns': List[str], 'metric_column': Optional[str], 'aggregation': 'SUM'|'AVG'|'COUNT'}
        - top_n_ranking: {'entity_column': str, 'metric_column': str, 'limit': int, 'order': 'DESC'|'ASC'}
        - period_over_period: {'date_column': str, 'metric_column': str, 'time_frame': 'MoM'|'YoY'}
        - cross_entity_performance: {'entity_column': str, 'metric_column': str}
        """
    )

# -----------------------------------------------------------------------------
# Heuristic Rule Engine (Fallback & Short-Circuit)
# -----------------------------------------------------------------------------
def heuristic_fallback(query: str, schema_columns: List[str]) -> AnalyticalPlan:
    """Provides rule-based query parsing when LLM calls hit rate limits or fail."""
    q_lower = query.lower()
    
    # 1. Filter out primary key / ID columns for grouping defaults
    non_id_cols = [c for c in schema_columns if not (c.lower().endswith("_id") or c.lower() == "id")]
    if not non_id_cols:
        non_id_cols = schema_columns

    # 2. Identify primary numeric metric column (revenue, sales, amount, total)
    metric_col = next(
        (c for c in schema_columns if any(m in c.lower() for m in ["revenue", "sales", "amount", "total", "price"])),
        schema_columns[-1]
    )

    # Row count queries
    if any(k in q_lower for k in ["how many records", "total rows", "row count", "total records"]):
        return AnalyticalPlan(tool_name=ToolName.GET_ROW_COUNT, tool_args={})

    # Period-over-period queries
    if any(k in q_lower for k in ["mom", "yoy", "month over month", "year over year", "growth", "trend", "monthly"]):
        date_col = next((c for c in schema_columns if any(d in c.lower() for d in ["date", "time", "month", "year", "created"])), schema_columns[0])
        time_frame = "YoY" if "year" in q_lower or "yoy" in q_lower else "MoM"
        return AnalyticalPlan(
            tool_name=ToolName.PERIOD_OVER_PERIOD,
            tool_args={"date_column": date_col, "metric_column": metric_col, "time_frame": time_frame}
        )

    # Cross-entity performance
    if any(k in q_lower for k in ["sales rep", "representative", "channel", "marketing", "campaign", "share"]):
        entity_col = next((c for c in schema_columns if any(e in c.lower() for e in ["rep", "channel", "agent", "seller", "campaign", "source"])), non_id_cols[0])
        return AnalyticalPlan(
            tool_name=ToolName.CROSS_ENTITY_PERFORMANCE,
            tool_args={"entity_column": entity_col, "metric_column": metric_col}
        )

    # Group By / Breakdown Analysis (e.g. "revenue by region", "sales by category")
    matched_dims = [c for c in non_id_cols if c.lower() in q_lower or any(w in q_lower for w in c.lower().split('_'))]
    
    return AnalyticalPlan(
        tool_name=ToolName.GROUP_BY_ANALYSIS,
        tool_args={
            "group_by_columns": matched_dims if matched_dims else [non_id_cols[0]],
            "metric_column": metric_col,
            "aggregation": "SUM"
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
        "You are an expert Data Analytics Router. Analyze the user query and schema columns, "
        "then select the best tool and parameters to answer the prompt.\n\n"
        "COLUMN SELECTION RULES:\n"
        "1. NEVER pick Primary Keys or Unique IDs (e.g., 'Transaction_ID', 'ID', 'Order_Number') as grouping columns unless explicitly asked.\n"
        "2. For queries asking for breakdowns 'by [X]' (e.g., 'revenue by region'), extract the categorical column matching [X] (e.g., 'Region').\n"
        "3. Match numeric metrics (e.g., 'revenue', 'sales', 'amount') to the relevant numeric column.\n\n"
        "Respond ONLY with a valid JSON object matching AnalyticalPlan."
    )
)

synthesizer_agent = Agent(
    model,
    system_prompt=(
        "You are a Senior Business Intelligence Lead. Given query results, provide a concise executive summary."
    )
)