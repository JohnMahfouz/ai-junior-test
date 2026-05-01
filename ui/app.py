import re
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import chat  # noqa: E402


def _dedupe_repeated_blocks(content: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    seen = set()
    unique_blocks = []
    for block in blocks:
        key = re.sub(r"\s+", " ", block).casefold()
        if key not in seen:
            seen.add(key)
            unique_blocks.append(block)
    return "\n\n".join(unique_blocks)


def _display_response(content: str) -> None:
    clean_content = re.sub(r"\[Source:.*?\]", "", content).strip()
    st.markdown(_dedupe_repeated_blocks(clean_content).replace("$", r"\$"))


st.set_page_config(page_title="NovaBite AI Assistant", page_icon="NB", layout="centered")
st.title("NovaBite AI Assistant")
st.caption("Ask about the menu, opening hours, policies, or make a booking.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _display_response(msg["content"])
        else:
            st.write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "agent_used": None})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chat(st.session_state.session_id, prompt)
        _display_response(result["response"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "agent_used": result["agent_used"],
    })
