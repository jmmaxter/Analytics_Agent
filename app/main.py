from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from app.agents.workflow import app_graph
# from app.api.routes import router
from app.api.routes import router as api_router

app = FastAPI(title="Analytics Agent", version="0.1.0")

# Enable Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits requests from any host (e.g. Render frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

class AnalyzeRequest(BaseModel):
    query: str
    dataset_path: str

@app.post("/query")
async def execute_query(payload: dict):
    # Your LangGraph / DuckDB / Gemini agent logic here
    return {"status": "success", "received_prompt": payload.get("prompt")}

@app.get("/")
def read_root():
    return {"status": "Analytics Agent is running"}

@app.post("/api/v1/analyze")
def run_analysis_workflow(payload: AnalyzeRequest):
    initial_state = {
        "query": payload.query,
        "dataset_path": payload.dataset_path,
        "errors": []
    }
    
    try:
        final_state = app_graph.invoke(initial_state)
        
        if final_state.get("errors"):
            raise HTTPException(status_code=400, detail=final_state["errors"])
            
        return {
            "status": final_state.get("status", "success"),
            "original_query": final_state.get("query"),
            "selected_tool": final_state.get("selected_tool"),
            "tool_args": final_state.get("tool_args"),
            "available_columns": final_state.get("available_columns", []),
            "insight": final_state.get("insight"),
            "raw_data": final_state.get("raw_data", [])
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))