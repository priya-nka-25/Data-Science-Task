"""
Streamlit Web UI for Task 2: Document-Aware Support Assistant.
Interactive multi-turn chat interface featuring:
- Session context tracking
- Anti-repetition enforcement
- Graceful topic transition notifications
- Section citations and source verification
- Knowledge base document viewer
"""

import os
import sys
import streamlit as st

st.set_page_config(
    page_title="Document-Aware Support Assistant",
    page_icon="🛡️",
    layout="wide"
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from task2_support_assistant.assistant import SupportAssistant

docs_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")

st.title("Document-Aware Enterprise Support Assistant")
st.caption("Context-Aware Support Assistant with Session Memory, Anti-Repetition, Graceful Topic Switching & Exact Section Citations")

# Sidebar
with st.sidebar:
    st.header(" Knowledge Base Documents")
    doc_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
    selected_doc = st.selectbox("Inspect Document", doc_files)
    if selected_doc:
        doc_path = os.path.join(docs_dir, selected_doc)
        with open(doc_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    st.divider()
    st.header(" 10-Turn Scripted Scenarios")
    scripted_turns = [
        "What are the response time commitments under our SLA for Critical and High severity incidents?",
        "What happens if a Critical incident is not resolved within 30 minutes? Who does it escalate to?",
        "Can you explain our refund policy for annual subscription contracts?",
        "Is there a 30-day money-back guarantee for first-time buyers?",
        "Going back to my very first question, what was the response time for Enterprise on critical outages?",
        "Switching topics, what are our API rate limits and token bucket quotas?",
        "What is our retry schedule if a webhook delivery fails?",
        "How do we handle Single Sign-On (SSO) and MFA authentication?",
        "What are the rules regarding password complexity and expiration intervals?",
        "Under GDPR, how long do we have to complete a customer data subject access request (DSAR)?"
    ]
    for i, t in enumerate(scripted_turns, 1):
        if st.button(f"Turn {i}: {t[:35]}...", key=f"t_{i}"):
            st.session_state["pending_input"] = t

    if st.button("🔄 Reset Conversation Session"):
        st.session_state["assistant"] = SupportAssistant(docs_dir)
        st.session_state["messages"] = []
        st.rerun()

# Initialize session state
if "assistant" not in st.session_state:
    st.session_state["assistant"] = SupportAssistant(docs_dir)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your Enterprise Document-Aware Support Assistant. I maintain session context, "
                "remember your previous queries, gracefully handle topic shifts, avoid repeating information, "
                "and strictly cite our policy sections for every answer. How can I assist you today?"
            ),
            "domain": "Greeting",
            "switched": False,
            "citations": None
        }
    ]

# Display message history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("switched"):
            st.warning(" **Topic Switch Detected & Gracefully Handled**")
        st.markdown(msg["content"])
        if msg.get("domain") and msg["role"] == "assistant" and msg["domain"] != "Greeting":
            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(f" **Domain:** `{msg['domain']}`")
            with col2:
                if msg.get("citations"):
                    st.caption(f" **Citation:** {msg['citations']}")

# Input handling
user_input = st.chat_input("Ask a question regarding SLAs, Billing, Security, APIs, or Privacy...")
if st.session_state.get("pending_input"):
    user_input = st.session_state.pop("pending_input")

if user_input:
    # Append user turn
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process via Assistant
    with st.chat_message("assistant"):
        with st.spinner("Retrieving document sections and evaluating conversational context..."):
            res = st.session_state["assistant"].answer(user_input)
            
            if res["is_topic_switched"]:
                st.warning("**Topic Switch Detected & Gracefully Handled**")
                
            st.markdown(res["assistant_response"])
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(f" **Domain:** `{res['topic_domain']}`")
            with col2:
                st.caption(f" **Citation:** {res['citations']}")

            st.session_state["messages"].append({
                "role": "assistant",
                "content": res["assistant_response"],
                "domain": res["topic_domain"],
                "switched": res["is_topic_switched"],
                "citations": res["citations"]
            })

