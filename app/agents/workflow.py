import os
import json
import pandas as pd
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from app.agents.analyzer import analytics_agent, synthesizer_agent, AnalyticalPlan, heuristic_fallback
from app.tools.registry import TOOL_REGISTRY

class AnalyticsState(TypedDict):
    query: str
    dataset_path: str
    schema_info: Dict[str, Any]
    selected_tool: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    available_columns: List[str]
    result_data: List[Dict[str, Any]]
    insight: Optional[str]
    status: Optional[str]
    errors: List[str]

def resolve_dataset_path(dataset_path: str) -> str:
    if os.path.exists(dataset_path):
        return dataset_path
    alt_paths = [
        dataset_path,
        os.path.join("app", dataset_path),
        os.path.join("..", dataset_path),
        os.path.basename(dataset_path),
        os.path.join("app", "data", os.path.basename(dataset_path))
    ]
    for path in alt_paths:
        if os.path.exists(path):
            return path
    return dataset_path

def node_analyze_query(state: AnalyticsState) -> Dict[str, Any]:
    query = state.get("query", "")
    raw_dataset_path = state.get("dataset_path", "app/data/data.csv")
    dataset_path = resolve_dataset_path(raw_dataset_path)
    
    default_columns = ["Transaction_ID", "Date", "Region", "Product_Category", "Revenue", "Units_Sold", "Discount_Applied"]
    try:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            columns = list(df.columns)
        else:
            columns = default_columns
    except Exception:
        columns = default_columns
    
    prompt = f"Dataset Columns: {columns}\nUser Query: {query}"
    
    tool_name = None
    tool_args = None
    
    try:
        result = analytics_agent.run_sync(prompt)
        raw_output = getattr(result, "data", getattr(result, "output", str(result)))

        # Safely parse JSON response into AnalyticalPlan
        if isinstance(raw_output, AnalyticalPlan):
            plan = raw_output
        elif isinstance(raw_output, dict):
            plan = AnalyticalPlan.model_validate(raw_output)
        elif isinstance(raw_output, str):
            cleaned_str = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            plan_dict = json.loads(cleaned_str)
            plan = AnalyticalPlan.model_validate(plan_dict)
        else:
            plan = None

        if plan and plan.tool_name:
            tool_name = plan.tool_name.value if hasattr(plan.tool_name, "value") else str(plan.tool_name)
            tool_args = plan.tool_args or {}

    except Exception as e:
        print(f"[ERROR] LLM routing agent exception: {str(e)}")
        
    if not tool_name or tool_name not in TOOL_REGISTRY:
        print(f"[INFO] Triggering heuristic fallback router for query: '{query}'")
        fallback_plan = heuristic_fallback(query, columns)
        tool_name = fallback_plan.tool_name.value if hasattr(fallback_plan.tool_name, "value") else str(fallback_plan.tool_name)
        tool_args = fallback_plan.tool_args
        
    print(f"[DEBUG] Final Routed Tool: {tool_name} with Args: {tool_args}")
    
    return {
        "selected_tool": tool_name,
        "tool_args": tool_args,
        "available_columns": columns,
        "dataset_path": dataset_path,
        "errors": []
    }

def node_execute_tool(state: AnalyticsState) -> Dict[str, Any]:
    if state.get("errors"): 
        return {}
        
    tool_name = state.get("selected_tool")
    args = state.get("tool_args", {})
    dataset_path = state.get("dataset_path")
    
    if not dataset_path or not os.path.exists(dataset_path):
        return {"errors": [f"Dataset file path '{dataset_path}' does not exist."]}
        
    if tool_name not in TOOL_REGISTRY:
        return {"errors": [f"Tool '{tool_name}' is not registered in TOOL_REGISTRY."]}
        
    try:
        executor_func = TOOL_REGISTRY[tool_name]
        records = executor_func(args, dataset_path)
        return {"result_data": records}
    except Exception as e:
        print(f"[ERROR] Tool execution exception: {str(e)}")
        return {"errors": [f"Execution error: {str(e)}"]}

def node_synthesize_insight(state: AnalyticsState) -> Dict[str, Any]:
    if state.get("errors"):
        return {
            "insight": f"Processing error: {state.get('errors')}", 
            "raw_data": [],
            "status": "error"
        }
        
    query = state.get("query")
    result_data = state.get("result_data", [])
    
    prompt = f"Original Query: {query}\nExecution Results: {result_data}"
    try:
        result = synthesizer_agent.run_sync(prompt)
        insight_text = getattr(result, "data", getattr(result, "output", str(result)))
    except Exception as e:
        insight_text = f"Retrieved {len(result_data)} rows successfully, but natural language synthesis failed: {str(e)}"
        
    return {
        "insight": insight_text,
        "raw_data": result_data,
        "status": "success"
    }

workflow = StateGraph(AnalyticsState)

workflow.add_node("analyze_query", node_analyze_query)
workflow.add_node("execute_tool", node_execute_tool)
workflow.add_node("synthesize_insight", node_synthesize_insight)

workflow.set_entry_point("analyze_query")
workflow.add_edge("analyze_query", "execute_tool")
workflow.add_edge("execute_tool", "synthesize_insight")
workflow.add_edge("synthesize_insight", END)

app_graph = workflow.compile()