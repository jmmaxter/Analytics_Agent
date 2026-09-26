from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agents.workflow import app_graph

app = FastAPI(title="Analytics Agent", version="0.1.0")

class AnalyzeRequest(BaseModel):
    query: str
    dataset_path: str

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
        # Invoke the compiled LangGraph workflow synchronously
        final_state = app_graph.invoke(initial_state)
        
        # Check if any node propagated an error
        if final_state.get("errors"):
            raise HTTPException(status_code=400, detail=final_state["errors"])
            
        return {
            "status": "success",
            "original_query": final_state["query"],
            "insight": final_state["final_insight"],
            "raw_data": final_state["result_data"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))