import streamlit as st
import requests
import pandas as pd
from typing import Optional

# Backend Configuration
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="Conversational Data Analytics Workspace",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Categorized Suggested Questions Map (Covers All Backend Tools)
# -----------------------------------------------------------------------------
SUGGESTED_QUESTIONS_BY_TOOL = {
    "📊 Dataset Scale": {
        "tool": "get_row_count",
        "queries": [
            "How many total records exist in this dataset?",
            "What is the total row count?"
        ]
    },
    "🔢 Metric Summaries": {
        "tool": "aggregate_metric",
        "queries": [
            "What is the grand total revenue?",
            "What is the average transaction value?"
        ]
    },
    "🔠 Breakdown & Grouping": {
        "tool": "group_by_analysis",
        "queries": [
            "What is the total revenue by region?",
            "What is the record count grouped by status?"
        ]
    },
    "🏆 Top N & Rankings": {
        "tool": "top_n_ranking",
        "queries": [
            "What are the top 3 product categories by units sold?",
            "Which top 5 regions generated the highest total profit?"
        ]
    },
    "📈 Period-over-Period": {
        "tool": "period_over_period",
        "queries": [
            "What is the Month-over-Month revenue growth?",
            "What is the Year-over-Year sales trend?"
        ]
    },
    "🤝 Cross-Entity Performance": {
        "tool": "cross_entity_performance",
        "queries": [
            "How do sales representatives compare in total revenue?",
            "What is the market share across sales channels?"
        ]
    }
}

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# -----------------------------------------------------------------------------
# Helper Functions for API Communication
# -----------------------------------------------------------------------------
def upload_file_to_backend(uploaded_file) -> Optional[dict]:
    """Sends uploaded file to FastAPI backend and retrieves session metadata."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/dataset/upload", files=files)
        if response.status_code == 201:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"
            st.error(f"Upload failed ({response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend server. Ensure FastAPI is running on port 8000.")
        return None

def submit_query_to_backend(session_id: str, query: str) -> Optional[dict]:
    """Sends natural language query to FastAPI endpoint for execution."""
    try:
        payload = {"session_id": session_id, "query": query}
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"
            st.error(f"Query error ({response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Connection error while reaching backend.")
        return None

def clear_backend_session(session_id: str) -> bool:
    """Notifies backend to purge temporary DuckDB database for this session."""
    try:
        response = requests.delete(f"{API_BASE_URL}/dataset/{session_id}")
        return response.status_code == 200
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Sidebar: Dataset Management & Upload
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📂 Dataset Workspace")
    
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload a dataset to begin asking natural language questions."
    )
    
    if uploaded_file and st.session_state.session_id is None:
        with st.spinner("Ingesting dataset and building DuckDB workspace..."):
            summary = upload_file_to_backend(uploaded_file)
            if summary:
                st.session_state.session_id = summary["session_id"]
                st.session_state.dataset_summary = summary
                st.success("Dataset loaded successfully!")
                st.rerun()

    # Active Session Information Card
    if st.session_state.session_id and st.session_state.dataset_summary:
        st.divider()
        st.markdown("### ℹ️ Dataset Profile")
        summary = st.session_state.dataset_summary
        
        st.metric("Filename", summary["filename"])
        col_a, col_b = st.columns(2)
        col_a.metric("Total Rows", f"{summary['total_rows']:,}")
        col_b.metric("Columns", len(summary["columns"]))
        
        st.caption(f"**Active Session ID:** `{st.session_state.session_id[:8]}...`")
        
        if st.button("🔴 Clear Dataset & Reset", use_container_width=True):
            if clear_backend_session(st.session_state.session_id):
                st.session_state.session_id = None
                st.session_state.dataset_summary = None
                st.session_state.query_history = []
                st.success("Workspace reset.")
                st.rerun()

# -----------------------------------------------------------------------------
# Main Workspace
# -----------------------------------------------------------------------------
st.title("🤖 Conversational Data Analyst")

if not st.session_state.session_id:
    st.info("👈 Please upload a CSV or Excel file in the sidebar to begin analysis.")
else:
    summary = st.session_state.dataset_summary

    # Data Schema Preview Collapsible
    with st.expander("🔍 Preview Dataset Schema & Sample Data", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Detected Columns & Types:**")
            schema_df = pd.DataFrame(
                list(summary["schema_types"].items()),
                columns=["Column", "Data Type"]
            )
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
            
        with col2:
            st.markdown("**First 5 Records:**")
            preview_df = pd.DataFrame(summary["preview_data"])
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    st.divider()

    # Tabbed Categorized Suggested Questions
    st.markdown("##### 💡 Suggested Questions by Analytics Category")
    
    selected_sample = None
    tab_labels = list(SUGGESTED_QUESTIONS_BY_TOOL.keys())
    tabs = st.tabs(tab_labels)

    for tab, (category_name, category_data) in zip(tabs, SUGGESTED_QUESTIONS_BY_TOOL.items()):
        with tab:
            queries = category_data["queries"]
            cols = st.columns(len(queries))
            for idx, q_text in enumerate(queries):
                if cols[idx].button(
                    q_text,
                    key=f"btn_{category_data['tool']}_{idx}",
                    use_container_width=True
                ):
                    selected_sample = q_text

    # Query Input Bar
    query_input = st.chat_input("Ask a question about your data...")
    active_query = selected_sample or query_input

    # Query Execution Flow
    if active_query:
        with st.spinner("Analyzing dataset with DuckDB and Gemini..."):
            result = submit_query_to_backend(st.session_state.session_id, active_query)
            if result:
                st.session_state.query_history.insert(0, result)

    # Render Results History Feed
    if st.session_state.query_history:
        st.markdown("### 📈 Analytical Output")
        
        for item in st.session_state.query_history:
            with st.container(border=True):
                st.subheader(f"❓ {item['original_query']}")
                
                # Render LLM Business Insight Narrative
                st.markdown(item["insight"])
                
                # Render Data Result Table (if results exist)
                if item["raw_data"]:
                    st.markdown("**Raw Query Output:**")
                    res_df = pd.DataFrame(item["raw_data"])
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                # Collapsible Technical Inspector
                with st.expander("⚙️ View Technical Details"):
                    t_col1, t_col2, t_col3 = st.columns(3)
                    t_col1.write(f"**Tool Used:** `{item['selected_tool']}`")
                    t_col2.write(f"**Execution Time:** `{item['execution_time_ms']} ms`")
                    t_col3.write(f"**Status:** `{item['status']}`")
                    st.json(item["tool_args"])