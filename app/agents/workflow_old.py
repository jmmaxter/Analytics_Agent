import os
import duckdb
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from app.data.ingestion import load_and_profile_data
from app.agents.analyzer import analytics_agent
from app.agents.analyzer import synthesis_agent

class AnalyticsState(TypedDict):
    """The state payload passed through the analytical graph."""
    query: str
    dataset_path: str        
    # Populated by the Ingestion Node
    dataset_profile: Optional[Dict[str, Any]]     
    # Populated by PydanticAI Reasoning Node
    selected_tool: Optional[str]        
    tool_args: Optional[Dict[str, Any]]     
    # Populated by the Execution Node
    result_data: Optional[Dict[str, Any]]     
    # Populated by the Final Synthesis Node
    final_insight: Optional[str]        
    errors: List[str]             

# --- Skeleton Nodes ---
def node_ingest_and_profile(state: AnalyticsState) -> Dict[str, Any]:
    """Reads the dataset and attaches the data profile to the state."""
    try:
        dataset_path = state.get("dataset_path")
        if not dataset_path or not os.path.exists(dataset_path):
             return {"errors": [f"Dataset not found at {dataset_path}"]}
             
        profile = load_and_profile_data(dataset_path)
        
        # Return a dict containing only the fields of state you want to update
        return {"dataset_profile": profile}
        
    except Exception as e:
        return {"errors": [f"Ingestion error: {str(e)}"]}

def node_plan_analytical_step(state: AnalyticsState) -> Dict[str, Any]:
    """Uses PydanticAI to map the query to a strict tool execution plan."""
    if state.get("errors"): 
        return {} # Short-circuit if previous node failed
        
    query = state.get("query")
    profile = state.get("dataset_profile")
    
    prompt = f"User Query: {query}\nDataset Schema: {profile}"
    
    try:
        # PydanticAI guarantees the response is a valid AnalyticalPlan model
        result = analytics_agent.run_sync(prompt)
        plan = result.output
        
        return {
            "selected_tool": plan.selected_tool,
            "tool_args": plan.tool_args
        }
    except Exception as e:
        return {"errors": [f"Planning error: {str(e)}"]}

def node_execute_tool(state: AnalyticsState) -> Dict[str, Any]:
    """Executes the selected analytical tool using DuckDB or Pandas."""
    if state.get("errors"): return {}
    
    tool_name = state.get("selected_tool")
    args = state.get("tool_args", {})
    dataset_path = state.get("dataset_path")
    
    try:
        if tool_name == "aggregate_metrics":
            # Gracefully handle key naming variations from LLM output
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
                return {"errors": [f"Missing grouping columns in tool arguments: {args}"]}
                
            if isinstance(group_keys, list):
                group_cols = ", ".join(group_keys)
            else:
                group_cols = str(group_keys)
                
            agg_col = (
                args.get("aggregate_column") or 
                args.get("column") or 
                args.get("metric") or 
                args.get("metric_column")
            )
            if not agg_col:
                return {"errors": [f"Missing aggregate column in tool arguments: {args}"]}
                
            func = str(
                args.get("aggregation_function") or 
                args.get("function") or 
                args.get("operation") or 
                "SUM"
            ).upper()
            
            # DuckDB natively reads the CSV and performs the aggregation
            query = f"""
                SELECT {group_cols}, {func}({agg_col}) as result 
                FROM read_csv_auto('{dataset_path}') 
                GROUP BY {group_cols}
            """
            result_df = duckdb.sql(query).df()
            
            return {"result_data": result_df.to_dict(orient="records")}
            
        else:
            return {"errors": [f"Tool '{tool_name}' execution is not implemented."]}
            
    except Exception as e:
        return {"errors": [f"Execution error: {str(e)}"]}

def node_synthesize_results(state: AnalyticsState) -> Dict[str, Any]:
    """Translates raw data outputs into a natural language insight."""
    if state.get("errors"): return {}
    
    query = state.get("query")
    raw_data = state.get("result_data")
    
    prompt = f"Original User Query: {query}\nRaw Data Result: {raw_data}"
    
    try:
        result = synthesis_agent.run_sync(prompt)
        return {"final_insight": result.output}
    except Exception as e:
        return {"errors": [f"Synthesis error: {str(e)}"]}

# Initialize Graph
workflow = StateGraph(AnalyticsState)
# --- Compile the Graph ---
workflow.add_node("ingest", node_ingest_and_profile)
workflow.add_node("plan", node_plan_analytical_step)
workflow.add_node("execute", node_execute_tool)
workflow.add_node("synthesize", node_synthesize_results)
# (Edges and routing logic would be mapped here)

# Define the synchronous linear flow
workflow.add_edge(START, "ingest")
workflow.add_edge("ingest", "plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "synthesize")
workflow.add_edge("synthesize", END)

# Export the compiled graph
app_graph = workflow.compile()