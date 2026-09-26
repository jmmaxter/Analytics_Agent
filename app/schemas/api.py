from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class DatasetSummary(BaseModel):
    session_id: str
    filename: str
    total_rows: int
    total_columns: int
    columns: List[str]
    schema_types: Dict[str, str]
    preview_data: List[Dict[str, Any]]

class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID from dataset upload")
    query: str = Field(..., description="Natural language question from the user")

class QueryResponse(BaseModel):
    status: str
    session_id: str
    original_query: str
    selected_tool: str
    tool_args: Dict[str, Any]
    insight: str
    raw_data: List[Dict[str, Any]]
    execution_time_ms: float