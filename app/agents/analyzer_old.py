import os
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Dict, Any

class AnalyticalPlan(BaseModel):
    selected_tool: str = Field(description="The name of the tool to use (e.g., 'aggregate_metrics').")
    tool_args: Dict[str, Any] = Field(description="The JSON arguments matching the tool's Pydantic schema.")

# Option A: Gemini (Best for complex reasoning and large schemas)
# Ensure GEMINI_API_KEY is in your .env
analytics_agent = Agent(
    'google:gemini-3.5-flash',
    output_type=AnalyticalPlan,
    system_prompt=(        
        "You are an analytical routing engine. Read the user's query and dataset profile. "
        "Available tools:\n"
        "1. 'aggregate_metrics': For calculating standard group-by metrics without sorting or limits.\n"
        "2. 'sort_and_rank': For queries asking for top/bottom items, rankings, or ordered extremes.\n"
        "3. 'filter_records': For queries requesting specific subsets or conditional row-level matches.\n"
        "4. 'clean_missing_values': For handling nulls, missing data, imputation, or dropping incomplete records.\n"
        "Map the user intent to the exact tool name and extract its parameters."
    )
)

synthesis_agent = Agent(
    'google:gemini-3.5-flash',
    output_type=str,
    system_prompt=(
        "You are an expert data analyst and business intelligence writer. "
        "Given the user's original natural language query and the raw execution data returned from the analytical tool, "
        "write a clear, professional, and well-structured natural language response that directly answers the user's question."
    )
)

# Option B: Groq (Best for blazing fast, simple tool routing)
# Ensure GROQ_API_KEY is in your .env
# analytics_agent = Agent(
#     'groq:llama-3.3-70b-versatile',
#     system_prompt="You are an analytical routing engine..."
# )