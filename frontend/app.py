import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Optional, Dict, Any, Tuple

# -----------------------------------------------------------------------------
# Streamlit Page Configuration (Must be set prior to other UI elements)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Conversational Data Analytics Workspace",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Backend URL Configuration (Render & Local Support)
# -----------------------------------------------------------------------------
def get_backend_urls() -> Tuple[str, str]:
    """
    Dynamically constructs the Base Host URL and API v1 Endpoint URL.
    Handles Render environment variables, protocol prepending, and trailing slashes.
    
    Returns:
        Tuple[str, str]: (base_host_url, api_v1_base_url)
    """
    raw_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip().rstrip("/")
    
    # If protocol is missing (e.g. Render 'host' property format)
    if not raw_url.startswith(("http://", "https://")):
        if "127.0.0.1" in raw_url or "localhost" in raw_url:
            raw_url = f"http://{raw_url}"
        else:
            raw_url = f"https://{raw_url}"
            
    # Strip /api/v1 suffix if present to get clean base host
    if raw_url.endswith("/api/v1"):
        base_host = raw_url[:-7]
    else:
        base_host = raw_url
        
    api_base_url = f"{base_host}/api/v1"
    return base_host, api_base_url

BASE_HOST_URL, API_BASE_URL = get_backend_urls()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "backend_online" not in st.session_state:
    st.session_state.backend_online = False

# -----------------------------------------------------------------------------
# Backend Health & Cold-Start Wake-Up Logic
# -----------------------------------------------------------------------------
def ensure_backend_healthy(max_wait_seconds: int = 60, poll_interval: int = 3) -> bool:
    """
    Pings backend endpoints to wake up sleeping services (Render Cold Start)
    or verify local connectivity before processing requests.
    """
    endpoints_to_try = [
        f"{BASE_HOST_URL}/health",
        f"{API_BASE_URL}/health",
        f"{BASE_HOST_URL}/"
    ]
    
    start_time = time.time()
    
    with st.spinner("⏳ Connecting to backend... (Waking up server if idle on Render)"):
        while (time.time() - start_time) < max_wait_seconds:
            for url in endpoints_to_try:
                try:
                    res = requests.get(url, timeout=5)
                    # 200 OK or 404 indicates the container is active and responding
                    if res.status_code in [200, 404]:
                        st.session_state.backend_online = True
                        return True
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    pass
            
            time.sleep(poll_interval)
            
    st.session_state.backend_online = False
    return False

# -----------------------------------------------------------------------------
# Helper Functions for API Communication
# -----------------------------------------------------------------------------
def upload_file_to_backend(uploaded_file) -> Optional[dict]:
    """Sends uploaded file to FastAPI backend and retrieves session metadata."""
    if not st.session_state.get("backend_online", False):
        if not ensure_backend_healthy():
            st.error(f"❌ Backend service at `{BASE_HOST_URL}` is unreachable. Ensure backend service is running.")
            return None

    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(
            f"{API_BASE_URL}/dataset/upload", 
            files=files,
            timeout=90  # Generous timeout for upload + DuckDB ingestion
        )
        if response.status_code == 201:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"
            st.error(f"Upload failed ({response.status_code}): {error_detail}")
            return None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.warning("⚠️ Connection interrupted. Attempting to wake up backend and retry upload...")
        if ensure_backend_healthy():
            return upload_file_to_backend(uploaded_file)
        st.error(f"Could not establish connection to backend at `{API_BASE_URL}`.")
        return None

def submit_query_to_backend(session_id: str, query: str) -> Optional[dict]:
    """Sends natural language query to FastAPI endpoint for execution."""
    if not st.session_state.get("backend_online", False):
        if not ensure_backend_healthy():
            st.error(f"❌ Backend service at `{BASE_HOST_URL}` is unreachable.")
            return None

    try:
        payload = {"session_id": session_id, "query": query}
        response = requests.post(
            f"{API_BASE_URL}/query", 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=90
        )
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"
            st.error(f"Query error ({response.status_code}): {error_detail}")
            return None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.warning("⚠️ Query timed out or lost connection. Waking backend and retrying...")
        if ensure_backend_healthy():
            return submit_query_to_backend(session_id, query)
        st.error("Request failed. Backend server may be offline or rebooting.")
        return None

def clear_backend_session(session_id: str) -> bool:
    """Notifies backend to purge temporary DuckDB database for this session."""
    try:
        response = requests.delete(f"{API_BASE_URL}/dataset/{session_id}", timeout=15)
        return response.status_code == 200
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Dynamic Visualization Renderer Engine
# -----------------------------------------------------------------------------
def render_analytics_chart(raw_data: List[Dict[str, Any]], selected_tool: str, tool_args: Dict[str, Any]):
    """
    Dynamically builds and renders interactive Plotly visualizations matching
    the analytical intent of the selected backend tool.
    """
    if not raw_data or len(raw_data) == 0:
        return

    df = pd.DataFrame(raw_data)
    cols = list(df.columns)

    if len(cols) == 0:
        return

    # 1. Single Value / Metric KPI
    if selected_tool in ["get_row_count", "aggregate_metric"] or len(df) == 1:
        if len(cols) == 1:
            val = df.iloc[0, 0]
            col_name = cols[0].replace("_", " ").title()
            val_str = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
            st.metric(label=col_name, value=val_str)
            return

    # Infer Column Categories
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    date_cols = [c for c in cols if any(k in c.lower() for k in ["date", "time", "year", "month", "day"])]

    # 2. Top N Ranking -> Horizontal Bar Chart
    if selected_tool == "top_n_ranking":
        y_col = cat_cols[0] if cat_cols else cols[0]
        x_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else cols[0])

        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            orientation="h",
            title=f"🏆 Top Ranking: {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
            color=x_col,
            color_continuous_scale="Viridis",
            text_auto=True
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        return

    # 3. Period-over-Period / Time Series -> Line Chart
    if selected_tool == "period_over_period" or (date_cols and num_cols):
        x_col = date_cols[0] if date_cols else (cat_cols[0] if cat_cols else cols[0])
        y_col = num_cols[0] if num_cols else cols[-1]

        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            markers=True,
            title=f"📈 Trend Analysis: {y_col.replace('_', ' ').title()} over {x_col.replace('_', ' ').title()}",
            template="plotly_white"
        )
        fig.update_traces(line_shape="linear", line=dict(width=3))
        fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        return

    # 4. Group-By Analysis -> Vertical Bar or Donut Chart
    if selected_tool == "group_by_analysis":
        x_col = cat_cols[0] if cat_cols else cols[0]
        y_col = num_cols[0] if num_cols else cols[-1]

        if len(df) <= 5:
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                hole=0.4,
                title=f"🍩 Distribution of {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
        else:
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=x_col,
                title=f"📊 Grouped Breakdown: {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                text_auto=True
            )
            fig.update_layout(xaxis_tickangle=-45)

        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        return

    # 5. Cross-Entity Performance -> Grouped Multi-Variable Bar
    if selected_tool == "cross_entity_performance" and len(cat_cols) >= 2:
        fig = px.bar(
            df,
            x=cat_cols[0],
            y=num_cols[0] if num_cols else cols[-1],
            color=cat_cols[1],
            barmode="group",
            title="🤝 Cross-Entity Performance Breakdown",
            text_auto=True
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        return

    # 6. Fallback Bar Chart
    if cat_cols and num_cols:
        fig = px.bar(
            df, 
            x=cat_cols[0], 
            y=num_cols[0], 
            title=f"📊 Visualizing {num_cols[0].replace('_', ' ').title()} by {cat_cols[0].replace('_', ' ').title()}", 
            text_auto=True
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Dynamic Suggested Questions Generator
# -----------------------------------------------------------------------------
def generate_suggested_questions(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates contextual suggested queries based on actual columns in the dataset."""
    if not summary:
        cat_col, cat_col2 = "category", "group"
        num_col = "value"
        date_col = "date"
    else:
        cols = summary.get("columns", [])
        schema = summary.get("schema_types", {})

        cat_cols = [c for c, dt in schema.items() if any(t in str(dt).upper() for t in ["VARCHAR", "TEXT", "STRING", "CHAR"])]
        num_cols = [c for c, dt in schema.items() if any(t in str(dt).upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT"])]
        date_cols = [c for c, dt in schema.items() if any(t in str(dt).upper() for t in ["DATE", "TIME", "TIMESTAMP"])]

        cat_col = cat_cols[0] if cat_cols else (cols[0] if cols else "category")
        cat_col2 = cat_cols[1] if len(cat_cols) > 1 else cat_col
        num_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else "value")
        date_col = date_cols[0] if date_cols else "date"

    return {
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
                f"What is the grand total {num_col}?",
                f"What is the average {num_col}?"
            ]
        },
        "🔠 Breakdown & Grouping": {
            "tool": "group_by_analysis",
            "queries": [
                f"What is the total {num_col} grouped by {cat_col}?",
                f"What is the record count grouped by {cat_col}?"
            ]
        },
        "🏆 Top N & Rankings": {
            "tool": "top_n_ranking",
            "queries": [
                f"What are the top 5 {cat_col} entries by total {num_col}?",
                f"Which top 3 {cat_col} entries generated the highest average {num_col}?"
            ]
        },
        "📈 Period-over-Period": {
            "tool": "period_over_period",
            "queries": [
                f"What is the Month-over-Month trend for {num_col} using {date_col}?",
                f"What is the Year-over-Year trend for total {num_col}?"
            ]
        },
        "🤝 Cross-Entity Performance": {
            "tool": "cross_entity_performance",
            "queries": [
                f"How do different {cat_col} values compare in total {num_col}?",
                f"What is the distribution of {num_col} across {cat_col2}?"
            ]
        }
    }

# -----------------------------------------------------------------------------
# Sidebar: Dataset Management & Backend Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📂 Dataset Workspace")
    
    # Backend Status Card
    st.markdown("### 🔌 Backend Connection")
    if st.session_state.backend_online:
        st.success(f"🟢 Active (`{BASE_HOST_URL}`)")
    else:
        st.warning(f"🔴 Offline / Sleeping (`{BASE_HOST_URL}`)")
        if st.button("⚡ Ping / Wake Up Backend", use_container_width=True):
            if ensure_backend_healthy():
                st.success("Backend is ready!")
                st.rerun()
            else:
                st.error("Could not reach backend server.")

    st.divider()

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

    # Dynamic Categorized Suggested Questions
    suggested_questions = generate_suggested_questions(summary)
    st.markdown("##### 💡 Suggested Questions (Contextualized to Your Dataset)")
    
    selected_sample = None
    tab_labels = list(suggested_questions.keys())
    tabs = st.tabs(tab_labels)

    for tab, (category_name, category_data) in zip(tabs, suggested_questions.items()):
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
                
                st.markdown(item["insight"])
                
                if item["raw_data"]:
                    st.markdown("**Raw Query Output:**")
                    res_df = pd.DataFrame(item["raw_data"])
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                with st.expander("⚙️ View Technical Details"):
                    t_col1, t_col2, t_col3 = st.columns(3)
                    t_col1.write(f"**Tool Used:** `{item['selected_tool']}`")
                    t_col2.write(f"**Execution Time:** `{item['execution_time_ms']} ms`")
                    t_col3.write(f"**Status:** `{item['status']}`")
                    st.json(item["tool_args"])