"""
Streamlit UI for the Price Comparison chatbot.
Powered by CrewAI agents + Grok (xAI).
"""

import os
import streamlit as st

# Load XAI_API_KEY from Streamlit secrets (for Streamlit Cloud deployment)
if "XAI_API_KEY" in st.secrets:
    os.environ["XAI_API_KEY"] = st.secrets["XAI_API_KEY"]

st.set_page_config(
    page_title="Quick Commerce Price Comparison",
    page_icon="🛒",
    layout="wide",
)

# ── Sidebar: API key configuration ──────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_key_input = st.text_input(
        "xAI API Key",
        value=os.environ.get("XAI_API_KEY", ""),
        type="password",
        placeholder="xai-...",
    )
    if api_key_input:
        os.environ["XAI_API_KEY"] = api_key_input

    st.markdown("---")
    st.markdown("**Platforms compared:**")
    st.markdown("🟢 Zepto\n🔵 Blinkit\n🟠 Swiggy Instamart")
    st.markdown("---")
    st.caption("Prices are fetched live from the web via DuckDuckGo search.")

# ── Main area ────────────────────────────────────────────────────────────────
st.title("🛒 Quick Commerce Price Comparison")
st.caption("Compare prices across **Zepto · Blinkit · Swiggy Instamart** in real time")

# Example prompts
st.markdown("**Try asking:**")
cols = st.columns(3)
examples = [
    "1 litre Amul milk",
    "Amul butter 500g",
    "Tropicana orange juice 1 litre",
]
for col, example in zip(cols, examples):
    if col.button(f"📦 {example}"):
        st.session_state["prefill"] = example

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (pre-fill from example buttons if clicked)
prefill = st.session_state.pop("prefill", "")
prompt = st.chat_input(
    "Ask me to compare prices… e.g. '1 litre Amul milk'",
) or prefill

if prompt:
    if not os.environ.get("XAI_API_KEY"):
        st.error("Please enter your xAI API key in the sidebar first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run CrewAI pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching prices across platforms… (this may take ~30 sec)"):
            try:
                from crew_agents import compare_prices
                result = compare_prices(prompt)
            except Exception as exc:
                result = f"❌ Error: {exc}"
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
