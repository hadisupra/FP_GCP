"""
Streamlit App for LLM Agent
Connects to FastAPI backend (local or GCP Cloud Run)
"""

import os
import json
from datetime import datetime

import requests
import streamlit as st

# Configuration - Set your API base URL (no trailing slash)
DEFAULT_API_URL = "https://llm-agent-api-793786022526.asia-southeast2.run.app"
API_URL = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")

# Page config
st.set_page_config(
    page_title="Olist LLM Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "auto"  # Default to auto mode
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    api_url_input = st.text_input(
        "API URL",
        value=API_URL,
        help="Your FastAPI backend URL (e.g., http://localhost:8080 or Cloud Run URL)",
    )
    if api_url_input:
        API_URL = api_url_input.rstrip("/")

    # Agent mode selector
    st.session_state.agent_mode = st.selectbox(
        "Agent Mode",
        options=["auto", "sql", "qdrant"],
        index=0,
        help="Auto: Let AI choose | SQL: Structured data | Qdrant: Product reviews"
    )

    st.divider()

    if st.button("🔌 Test Connection", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected to API!")
                data = response.json()
                st.json(data)
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection failed: {str(e)}")

    st.divider()

    st.subheader("📊 Session Info")
    st.text(f"Session ID: {st.session_state.session_id}")
    st.text(f"Messages: {len(st.session_state.messages)}")
    st.text(f"Current Mode: {st.session_state.agent_mode.upper()}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    
# Main content
st.title("🤖 Olist E-Commerce Intelligence Agent")
st.markdown("""
### 💬 Chat with your data using AI
Ask questions about products, orders, reviews, and get intelligent insights powered by dual RAG agents.
""")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show metadata if exists
        if message["role"] == "assistant" and "metadata" in message and message["metadata"]:
            metadata = message["metadata"]
            if "agents" in metadata and metadata["agents"]:
                st.caption(f"🤖 {', '.join(metadata['agents'])}")
            if "agent_choice" in metadata:
                st.caption(f"🎯 Routed to: {metadata['agent_choice']}")

# Process pending message from Quick Start buttons
if st.session_state.pending_message:
    prompt = st.session_state.pending_message
    st.session_state.pending_message = None
else:
    prompt = None

# Chat input and handling
if not prompt:
    prompt = st.chat_input("💭 Ask me anything about products, orders, or reviews...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Send prompt to API and render response
    try:
        payload = {
            "message": prompt,
            "agent": st.session_state.agent_mode,
            "session_id": st.session_state.session_id,
        }

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question..."):
                resp = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    timeout=60,
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("agent_response", "No response received")
                    agents_used = data.get("agents_used", [])
                    agent_choice = data.get("agent_choice", "unknown")

                    st.markdown(answer)

                    if agents_used:
                        st.caption(f"🤖 Agents: {', '.join(agents_used)}")
                    if agent_choice and st.session_state.agent_mode == "auto":
                        st.caption(f"🎯 Auto-routed to: **{agent_choice.upper()}**")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "agents": agents_used,
                            "agent_choice": agent_choice,
                            "mode": st.session_state.agent_mode,
                        },
                    })
                else:
                    try:
                        error_detail = resp.json()
                        error_body = json.dumps(error_detail, indent=2)
                        error_text = f"❌ API Error {resp.status_code}\n\n```json\n{error_body}\n```"
                    except Exception:
                        error_text = f"❌ API Error {resp.status_code}\n\n{resp.text}"

                    st.error(error_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_text,
                        "metadata": None,
                    })
    except requests.exceptions.Timeout:
        error_msg = "⏱️ **Request Timeout**\n\nThe request took too long. The API might be processing a complex query or experiencing high load. Please try again."
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "metadata": None,
        })
    except requests.exceptions.ConnectionError:
        error_msg = f"🔌 **Connection Error**\n\nCannot reach the API at `{API_URL}`\n\nPlease check:\n- API URL is correct\n- API service is running\n- Network connection"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "metadata": None,
        })
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ **Request Error**\n\n{str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "metadata": None,
        })
    except Exception as e:
        error_msg = f"⚠️ **Unexpected Error**\n\n{str(e)}\n\nPlease report this issue."
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "metadata": None,
        })

# Footer
st.divider()

# Quick action buttons (only show if no active chat)
if len(st.session_state.messages) == 0:
    st.subheader("🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Product Statistics", use_container_width=True):
            st.session_state.pending_message = "Berapa total produk yang dijual?"
            st.rerun()
    
    with col2:
        if st.button("⭐ Review Analysis", use_container_width=True):
            st.session_state.pending_message = "Show me customer review summary for perfume products"
            st.rerun()
    
    with col3:
        if st.button("💰 Price Analysis", use_container_width=True):
            st.session_state.pending_message = "What is the average price of products by category?"
            st.rerun()

st.divider()
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>Powered by FastAPI + SQLite + Qdrant + OpenAI | Deployed on GCP Cloud Run</small>
</div>
""",
    unsafe_allow_html=True,
)
