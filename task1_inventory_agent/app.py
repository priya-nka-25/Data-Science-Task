"""
Streamlit Chat Application for Task 1: Inventory Agent.
Provides an interactive chat interface allowing users to:
- Query Excel inventory data via natural language
- View dynamic Python code execution and results
- View domain definition lookups
- Read plain English summaries with business recommendations
- Explore interactive charts and summary tables
"""

import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Inventory AI Agent",
    page_icon="📦",
    layout="wide"
)

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from task1_inventory_agent.agent import InventoryAgent

st.title("Inventory Data Intelligence Agent")
st.caption("AI-Powered Agent for Excel Data Querying, Domain Definition Search & Plain-English Synthesis")

# Sidebar Configuration
with st.sidebar:
    st.header(" Configuration & Data")
    
    # API Key inputs (optional)
    api_provider = st.selectbox("LLM Provider Mode", ["Offline / Semantic Planner (No Key Required)", "Google Gemini", "OpenAI"])
    gemini_key = None
    openai_key = None
    if api_provider == "Google Gemini":
        gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    elif api_provider == "OpenAI":
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    # Dataset selector / uploader
    default_excel = os.path.join(os.path.dirname(__file__), "data", "Inventory-Records-Sample-Data.xlsx")
    if not os.path.exists(default_excel):
        default_excel = r"Question 1 & 3 (Files to use)\Question 1 & 3 (Files to use)\Question 1\Inventory-Records-Sample-Data.xlsx"

    uploaded_file = st.file_uploader("Upload Inventory Excel (.xlsx)", type=["xlsx", "xls"])
    if uploaded_file:
        temp_path = os.path.join(os.path.dirname(__file__), "temp_uploaded.xlsx")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_dataset_path = temp_path
        st.success("Uploaded custom dataset loaded!")
    else:
        active_dataset_path = default_excel
        st.info("Using default sample dataset (`Inventory-Records-Sample-Data.xlsx`)")

    # Quick Prompts
    st.subheader(" Quick Prompts")
    quick_prompts = [
        "What are the top 5 products by total inventory valuation?",
        "What does Inventory Turnover Ratio mean and which products have the highest turnover?",
        "Which products have low stock or high stockout risk?",
        "Give me an executive summary of total inventory value, total units sold, and stock on hand."
    ]
    for qp in quick_prompts:
        if st.button(qp, key=f"btn_{qp[:15]}"):
            st.session_state["pending_prompt"] = qp

# Initialize Agent in session state
if "agent" not in st.session_state or st.session_state.get("current_path") != active_dataset_path:
    st.session_state["agent"] = InventoryAgent(
        active_dataset_path, 
        gemini_api_key=gemini_key, 
        openai_api_key=openai_key
    )
    st.session_state["current_path"] = active_dataset_path

agent: InventoryAgent = st.session_state["agent"]

# Dataset Overview Metrics
with st.expander(" Dataset Overview & Preview", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    df = agent.df
    col1.metric("Total SKUs", len(df))
    col2.metric("Total Units On Hand", f"{int(df['Hand_In_Stock'].sum()):,}")
    col3.metric("Total Units Sold", f"{int(df['Units_Sold'].sum()):,}")
    col4.metric("Inventory Valuation", f"${float(df['Cost_Price_Total_USD'].sum()):,.2f}")
    st.dataframe(df, use_container_width=True)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your Inventory AI Data Agent. I can write and execute code to answer questions "
                "about your inventory dataset, look up supply chain definitions, calculate operational KPIs "
                "(turnover ratio, stockout risk, holding costs), and summarize everything in plain English.\n\n"
                "Try asking me about top products by value, high turnover items, or supply chain metrics!"
            )
        }
    ]

# Render Chat History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            det = msg["details"]
            if det.get("search_context"):
                sc = det["search_context"]
                st.info(f"🔍 **Search Definition: {sc.get('matched_term').title()}**\n\n{sc.get('definition')}")
            
            with st.expander("🛠️ View Executed Code & REPL Output"):
                st.code(det.get("code", ""), language="python")
                raw_res = det.get("execution", {}).get("result_raw")
                if isinstance(raw_res, pd.DataFrame):
                    st.dataframe(raw_res)
                    # Show plot if suitable
                    if len(raw_res) > 1 and "Product_Name" in raw_res.columns:
                        numeric_cols = raw_res.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            fig = px.bar(
                                raw_res, 
                                x="Product_Name", 
                                y=numeric_cols[-1], 
                                title=f"{numeric_cols[-1]} by Product",
                                color=numeric_cols[-1]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                elif raw_res is not None:
                    st.write(raw_res)

# Process Prompt
user_prompt = st.chat_input("Ask a question about the inventory data...")
if st.session_state.get("pending_prompt"):
    user_prompt = st.session_state.pop("pending_prompt")

if user_prompt:
    # Append User Message
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query, executing code and retrieving context..."):
            result = agent.ask(user_prompt)
            st.markdown(result["summary"])
            
            # Interactive details
            if result.get("search_context"):
                sc = result["search_context"]
                st.info(f" **Search Definition: {sc.get('matched_term').title()}**\n\n{sc.get('definition')}")
            
            with st.expander("🛠️ View Executed Code & REPL Output"):
                st.code(result.get("code", ""), language="python")
                raw_res = result.get("execution", {}).get("result_raw")
                if isinstance(raw_res, pd.DataFrame):
                    st.dataframe(raw_res)
                    if len(raw_res) > 1 and "Product_Name" in raw_res.columns:
                        numeric_cols = raw_res.select_dtypes(include="number").columns
                        if len(numeric_cols) > 0:
                            fig = px.bar(
                                raw_res, 
                                x="Product_Name", 
                                y=numeric_cols[-1], 
                                title=f"{numeric_cols[-1]} by Product",
                                color=numeric_cols[-1]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                elif raw_res is not None:
                    st.write(raw_res)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["summary"],
                "details": result
            })

