import os
import io
import time
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Optional, Dict, Any, Tuple, List

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

def upload_multi_files_to_backend(uploaded_files) -> Optional[dict]:
    """Sends multiple files to FastAPI backend for multi-table DuckDB ingestion."""
    if not st.session_state.get("backend_online", False):
        if not ensure_backend_healthy():
            st.error(f"❌ Backend service at `{BASE_HOST_URL}` is unreachable.")
            return None

    try:
        files = [
            ("files", (f.name, f.getvalue(), f.type))
            for f in uploaded_files
        ]
        response = requests.post(
            f"{API_BASE_URL}/dataset/upload-multi", 
            files=files,
            timeout=120
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
        st.error("Could not establish connection to backend for multi-file upload.")
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
def render_analytics_chart(raw_data: Any, selected_tool: str, tool_args: Dict[str, Any], chart_uid: str = "0"):
    """
    Safely parses raw data, automatically selects the most readable Plotly chart type
    based on data shape & cardinality (e.g., Horizontal Top-N Bar or Treemap for many categories,
    Line Chart for time series), provides an interactive chart-type switcher, and supports HTML export.
    """
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except Exception:
            return

    if not raw_data:
        return

    try:
        df = pd.DataFrame(raw_data)
    except Exception as e:
        st.error(f"Error creating DataFrame for chart: {e}")
        return

    if df.empty or len(df.columns) == 0:
        return

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "year", "month", "day", "period"])]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]

    if not cat_cols and not date_cols and len(df.columns) >= 1:
        cat_cols = [df.columns[0]]

    # Multi-metric handling
    if len(num_cols) > 1 and cat_cols:
        st.markdown("### 📊 Multi-Metric Visual Breakdown")
        for idx, num_col in enumerate(num_cols):
            plot_df = df.sort_values(by=num_col, ascending=False).head(15) if len(df) > 15 else df
            is_high_card = len(df) > 12
            if is_high_card:
                fig = px.bar(
                    plot_df.iloc[::-1],
                    x=num_col,
                    y=cat_cols[0],
                    orientation="h",
                    color=num_col,
                    title=f"📊 Top {len(plot_df)} {cat_cols[0].replace('_', ' ').title()} by {num_col.replace('_', ' ').title()}",
                    text_auto=True,
                    template="plotly_white",
                    color_continuous_scale=px.colors.sequential.Viridis
                )
            else:
                fig = px.bar(
                    plot_df,
                    x=cat_cols[0],
                    y=num_col,
                    color=cat_cols[0],
                    title=f"📊 {num_col.replace('_', ' ').title()} by {cat_cols[0].replace('_', ' ').title()}",
                    text_auto=True,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig.update_layout(showlegend=False)

            fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
            plotly_config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "filename": f"analytics_chart_{num_col}_{int(time.time())}", "scale": 2}
            }
            chart_col, export_col = st.columns([4, 1])
            with chart_col:
                st.plotly_chart(fig, use_container_width=True, config=plotly_config, key=f"chart_{chart_uid}_{idx}")
            with export_col:
                st.markdown("<br>", unsafe_allow_html=True)
                html_data = fig.to_html(include_plotlyjs="cdn")
                st.download_button(
                    label="📥 Export HTML",
                    data=html_data,
                    file_name=f"chart_{num_col}_{int(time.time())}.html",
                    mime="text/html",
                    use_container_width=True,
                    key=f"dl_{chart_uid}_{num_col}"
                )
        return

    # Single-metric adaptive chart selection
    metric_col = num_cols[0] if num_cols else (df.columns[1] if len(df.columns) >= 2 else None)
    primary_cat = cat_cols[0] if cat_cols else None
    secondary_cat = cat_cols[1] if len(cat_cols) >= 2 else None
    primary_date = date_cols[0] if date_cols else None

    if not metric_col:
        return

    row_count = len(df)
    unique_cats = df[primary_cat].nunique() if primary_cat else 0

    # Determine optimal default visualization based on data shape
    if primary_date and primary_cat:
        default_chart = "Line Trend"
    elif primary_date:
        default_chart = "Line Trend"
    elif unique_cats > 12:
        default_chart = "Horizontal Bar (Top N)"
    elif unique_cats <= 6 and not secondary_cat:
        default_chart = "Donut / Pie"
    else:
        default_chart = "Vertical Bar"

    chart_options = ["Auto (Recommended)", "Horizontal Bar (Top N)", "Treemap (All Categories)", "Vertical Bar", "Line Trend", "Donut / Pie"]
    ctrl_col1, ctrl_col2 = st.columns([2, 2])
    with ctrl_col1:
        chosen_mode = st.selectbox(
            f"Chart Visualization (Recommended: {default_chart})",
            chart_options,
            index=0,
            key=f"chart_mode_{chart_uid}"
        )
    active_chart = default_chart if chosen_mode == "Auto (Recommended)" else chosen_mode

    top_n = row_count
    if active_chart == "Horizontal Bar (Top N)" and row_count > 10:
        with ctrl_col2:
            top_n = st.slider(
                "Show Top N Items",
                min_value=5,
                max_value=min(50, row_count),
                value=min(15, row_count),
                step=5,
                key=f"top_n_{chart_uid}"
            )

    fig = None

    if active_chart == "Horizontal Bar (Top N)" and primary_cat:
        plot_df = df.sort_values(by=metric_col, ascending=False).head(top_n)
        # Reverse order so highest bar renders at the top
        plot_df_asc = plot_df.iloc[::-1]
        title_prefix = f"Top {len(plot_df)} of {unique_cats}" if unique_cats > top_n else f"All {unique_cats}"
        fig = px.bar(
            plot_df_asc,
            x=metric_col,
            y=primary_cat,
            orientation="h",
            color=metric_col,
            title=f"📊 {metric_col.replace('_', ' ').title()} by {primary_cat.replace('_', ' ').title()} ({title_prefix})",
            text_auto=True,
            template="plotly_white",
            color_continuous_scale=px.colors.sequential.Tealgrn
        )
        fig.update_layout(height=max(420, len(plot_df) * 28), coloraxis_showscale=False)

    elif active_chart == "Treemap (All Categories)" and primary_cat:
        path_cols = [primary_cat, secondary_cat] if secondary_cat else [primary_cat]
        plot_df = df[df[metric_col] > 0]
        if not plot_df.empty:
            fig = px.treemap(
                plot_df,
                path=path_cols,
                values=metric_col,
                color=metric_col,
                title=f"🗺️ Proportional Treemap: {metric_col.replace('_', ' ').title()} across {primary_cat.replace('_', ' ').title()}",
                template="plotly_white",
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig.update_traces(textinfo="label+value+percent root")

    elif active_chart == "Line Trend" and (primary_date or primary_cat):
        x_axis = primary_date if primary_date else primary_cat
        plot_df = df.sort_values(by=x_axis)
        if primary_date and primary_cat:
            # Limit legend clutter to top 8 categories by total metric
            top_entities = (
                plot_df.groupby(primary_cat)[metric_col]
                .sum()
                .nlargest(8)
                .index.tolist()
            )
            plot_df = plot_df[plot_df[primary_cat].isin(top_entities)]
            fig = px.line(
                plot_df,
                x=x_axis,
                y=metric_col,
                color=primary_cat,
                markers=True,
                title=f"📈 Trend of {metric_col.replace('_', ' ').title()} over {x_axis.replace('_', ' ').title()} (Top {len(top_entities)})",
                template="plotly_white"
            )
        else:
            fig = px.line(
                plot_df,
                x=x_axis,
                y=metric_col,
                markers=True,
                title=f"📈 Trend of {metric_col.replace('_', ' ').title()} over {x_axis.replace('_', ' ').title()}",
                template="plotly_white"
            )
            fig.update_traces(line=dict(width=3, color="#636EFA"))

    elif active_chart == "Donut / Pie" and primary_cat:
        plot_df = df.sort_values(by=metric_col, ascending=False).head(15)
        fig = px.pie(
            plot_df,
            names=primary_cat,
            values=metric_col,
            hole=0.45,
            title=f"🍩 Share of {metric_col.replace('_', ' ').title()} by {primary_cat.replace('_', ' ').title()}",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")

    else:
        # Vertical Bar fallback
        x_col = primary_cat or primary_date or df.columns[0]
        if secondary_cat:
            fig = px.bar(
                df.head(30),
                x=x_col,
                y=metric_col,
                color=secondary_cat,
                barmode="group",
                title=f"📊 {metric_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()} & {secondary_cat.replace('_', ' ').title()}",
                text_auto=True,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
        else:
            fig = px.bar(
                df.head(30),
                x=x_col,
                y=metric_col,
                color=x_col,
                title=f"📊 {metric_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                text_auto=True,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_tickangle=-35 if len(df) > 5 else 0)

    if fig is not None:
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        
        plotly_config = {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"analytics_chart_{int(time.time())}",
                "scale": 2
            }
        }

        chart_col, export_col = st.columns([4, 1])
        with chart_col:
            st.plotly_chart(fig, use_container_width=True, config=plotly_config, key=f"plotly_{chart_uid}")
        with export_col:
            st.markdown("<br>", unsafe_allow_html=True)
            html_data = fig.to_html(include_plotlyjs="cdn")
            st.download_button(
                label="📥 Export HTML",
                data=html_data,
                file_name=f"chart_{int(time.time())}.html",
                mime="text/html",
                help="Download interactive HTML dashboard component",
                use_container_width=True,
                key=f"export_btn_{chart_uid}"
            )
            st.caption("📄 **PDF Tip:** Click **Export HTML**, open the file in your browser, and press `Ctrl+P` (or `Cmd+P`) → **Save as PDF**.")

# -----------------------------------------------------------------------------
# Dynamic Suggested Questions Generator (Schema-Agnostic Taxonomy)
# -----------------------------------------------------------------------------
def generate_suggested_questions(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates contextual suggested queries based on actual columns in the dataset using the refined taxonomy."""
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
        "📊 Metric Summaries": {
            "tool": "aggregate_metric",
            "queries": [
                f"What is the total record count and summary of available metrics for {num_col}?",
                f"Show me the average, minimum, and maximum values for {num_col}."
            ]
        },
        "📂 Breakdown & Grouping": {
            "tool": "group_by_analysis",
            "queries": [
                f"Break down {num_col} by {cat_col}.",
                f"Give me a multi-level performance breakdown of {num_col} grouped by {cat_col} and {cat_col2}."
            ]
        },
        "🏆 Top N & Ranking": {
            "tool": "top_n_ranking",
            "queries": [
                f"What are the top 5 records ranked by {num_col} in descending order?",
                f"Show me the lowest-performing {cat_col} based on {num_col}."
            ]
        },
        "📈 Period-over-Period": {
            "tool": "period_over_period",
            "queries": [
                f"Show me the trend of {num_col} over time by {date_col}.",
                f"How has {num_col} progressed across available time intervals?"
            ]
        },
        "🔗 Cross-Entity Performance": {
            "tool": "cross_entity_performance",
            "queries": [
                f"How do {cat_col} and {cat_col2} interact regarding {num_col} performance?",
                f"Compare metrics across {cat_col} and {cat_col2} simultaneously."
            ]
        },
        "🔍 Data Quality & Profiling": {
            "tool": "data_profiling",
            "queries": [
                "Check for missing values, null rates, and data completeness across all columns.",
                "Profile the dataset schema to identify unique value counts, data types, and potential anomalies."
            ]
        },
        "🥧 Share of Total & Contribution": {
            "tool": "share_of_total",
            "queries": [
                f"What is the percentage contribution (share of total) of each {cat_col} to the overall {num_col}?",
                f"Show the relative proportion and weight of each {cat_col} against the aggregate sum of {num_col}."
            ]
        }
    }

# -----------------------------------------------------------------------------
# Comprehensive Analytics & BI Question Catalog
# -----------------------------------------------------------------------------
BI_QUESTION_CATALOG = {
    "📊 Metric Summaries": [
        "What is the total revenue and overall profit across all records?",
        "Show me total units sold, total revenue, and average unit cost.",
        "What is the total discount applied and average order transaction value?",
        "Provide a summary of total revenue, profit margins, and average customer ratings."
    ],
    "🔍 Breakdown & Grouping": [
        "What is total sales by product category and region?",
        "Show me total revenue and profit broken down by customer segment and payment method.",
        "What is the average transaction value and total units sold by store location?",
        "Show total shipping cost and average transit time by hub region.",
        "Break down revenue by marketing channel and sales rep."
    ],
    "🏆 Top N & Ranking": [
        "Show the top 5 product categories by total revenue.",
        "What are the top 3 store locations by profit?",
        "List the top 10 transactions by revenue amount.",
        "Show the top 5 sales representatives by total units sold.",
        "What are the bottom 3 product categories by profit margin?"
    ],
    "📈 Period-over-Period": [
        "Calculate Month-over-Month revenue growth.",
        "Show Year-over-Year performance trends for total profit.",
        "What is the quarterly breakdown of sales and transaction counts?",
        "Compare monthly revenue against the previous period across all regions."
    ],
    "💼 Cross-Entity Performance": [
        "Analyze transaction performance and average rating by customer segment.",
        "Show sales performance and transaction count by marketing channel.",
        "Compare store square footage efficiency against total revenue generated.",
        "Evaluate sales rep performance across different store locations."
    ],
    "🧹 Data Quality & Profiling": [
        "Check for missing values or null records across the active dataset.",
        "Show record counts, unique values, and summary statistics for key columns.",
        "Identify any anomalous transactions with zero or negative revenue.",
        "Verify referential integrity and missing foreign keys across star schema tables."
    ],
    "🥧 Share of Total & Contribution": [
        "What percentage of total profit does each customer segment contribute?",
        "Show the revenue contribution percentage of each product category.",
        "What share of total sales does each store region account for?",
        "Calculate the percentage contribution of each marketing channel to total revenue."
    ]
}

# -----------------------------------------------------------------------------
# Sidebar: Dataset Management, Backend Status & BI Catalog
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

    # Workspace Mode Selector
    workspace_mode = st.radio(
        "Select Workspace Mode:",
        ["Single Dataset", "Enterprise BI Project (Multi-Table)"],
        help="Choose Single Dataset for fast ad-hoc queries or Multi-Table for Star Schema relational projects."
    )

    st.divider()

    # Conditional Upload Flow based on Workspace Mode
    if workspace_mode == "Single Dataset":
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
    else:
        uploaded_files = st.file_uploader(
            "Upload Fact & Dimension Tables (Multiple)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload multiple related tables (e.g., Fact transactions and Dim products)."
        )
        
        if uploaded_files and st.session_state.session_id is None:
            with st.spinner("Ingesting BI Project tables into DuckDB workspace..."):
                summary = upload_multi_files_to_backend(uploaded_files)
                if summary:
                    st.session_state.session_id = summary["session_id"]
                    st.session_state.dataset_summary = summary
                    st.success("BI Project tables loaded successfully!")
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
        
        # --- BI Question Catalog Section (Enabled when dataset is active) ---
        st.divider()
        st.markdown("### 📚 BI Question Catalog")
        st.caption("Select a category and click any professional query to run it instantly.")
        
        selected_category = st.selectbox("Analysis Category", list(BI_QUESTION_CATALOG.keys()), key="sidebar_catalog_category")
        
        if selected_category:
            for question in BI_QUESTION_CATALOG[selected_category]:
                if st.button(question, key=f"cat_btn_{hash(question)}", use_container_width=True):
                    st.session_state.preset_query = question
                    st.rerun()

        st.divider()
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

    # Dynamic Categorized Suggested Questions (Schema-Agnostic Taxonomy)
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

    # -------------------------------------------------------------------------
    # Render Results History Feed
    # -------------------------------------------------------------------------
    if st.session_state.query_history:
        st.markdown("### 📈 Analytical Output")
        
        for idx, item in enumerate(st.session_state.query_history):
            with st.container(border=True):
                st.subheader(f"❓ {item['original_query']}")
                
                # 1. Executive Summary / Text Insight
                st.markdown(item["insight"])
                
                # =============================================================
                # 🐛 DEBUG BLOCK — Place right here to inspect backend payload
                # =============================================================
                with st.expander("🐛 Debug Response Payload", expanded=True):
                    st.write("**`selected_tool`:**", item.get("selected_tool"))
                    st.write("**`raw_data` type:**", type(item.get("raw_data")))
                    st.write("**`raw_data` content:**", item.get("raw_data"))
                # =============================================================
                
                # 2. Dynamic Chart Visualization
                raw_data = item.get("raw_data")
                
                # Handle stringified JSON if backend returns string instead of list
                if isinstance(raw_data, str):
                    try:
                        raw_data = json.loads(raw_data)
                    except Exception:
                        raw_data = None

                if raw_data:
                    render_analytics_chart(
                        raw_data=raw_data,
                        selected_tool=item.get("selected_tool", ""),
                        tool_args=item.get("tool_args", {}),
                        chart_uid=f"hist_{idx}"
                    )
                else:
                    st.warning("⚠️ No tabular `raw_data` received in payload to render a chart.")

                
                # 3. Tabular Data View
                if raw_data:
                    with st.expander("📋 View Raw Data Table"):
                        res_df = pd.DataFrame(raw_data)
                        st.dataframe(res_df, use_container_width=True, hide_index=True)