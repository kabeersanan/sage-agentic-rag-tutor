import os
from pathlib import Path
import streamlit as st
import requests

st.set_page_config(page_title="AI Tutor | Class 10 Science", layout="wide")

st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .source-box {font-size: 0.8em; color: #666; background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)
 
# Backend base URL, resolved in priority order:
#   1. BACKEND_URL env var (Render, or a local override)
#   2. st.secrets["BACKEND_URL"] (Streamlit Community Cloud)
#   3. 127.0.0.1 local fallback — use 127.0.0.1 not "localhost" to match
#      uvicorn's IPv4 bind; on Windows "localhost" can resolve to IPv6 (::1)
#      and fail to connect even when the backend is running fine.
# We only read st.secrets if a secrets.toml actually exists, because accessing
# st.secrets with no file present makes Streamlit render a "No secrets files
# found" error in the UI (a caught exception doesn't suppress that display).
def _resolve_backend():
    env_url = os.environ.get("BACKEND_URL")
    if env_url:
        return env_url
    secret_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path(".streamlit") / "secrets.toml",
    ]
    if any(p.exists() for p in secret_paths):
        try:
            return st.secrets["BACKEND_URL"]
        except Exception:
            pass
    return "http://127.0.0.1:8000"

BACKEND = _resolve_backend()
API_URL = f"{BACKEND}/query"
UPLOAD_URL = f"{BACKEND}/upload"
KB_URL = f"{BACKEND}/knowledge-base"
HEALTH_URL = f"{BACKEND}/health"

with st.sidebar:
    st.title("AI Tutor Settings")
    st.divider()
    st.markdown("**System Status**")

    # Connectivity is decided by /health, which is fast (~0.2s). The chapter
    # list comes from /knowledge-base, which is slower (it queries Qdrant), so we
    # fetch it separately with a generous timeout. This way a slow stats call can
    # never make a healthy backend look "offline".
    backend_up = False
    try:
        requests.get(HEALTH_URL, timeout=5).raise_for_status()
        backend_up = True
        st.text("Backend: Connected")
    except requests.exceptions.RequestException:
        st.error("Backend offline — start it on 127.0.0.1:8000.")

    if backend_up:
        try:
            kb = requests.get(KB_URL, timeout=15).json()
            docs = kb.get("documents", [])
            if docs:
                st.success(f"Chapters loaded: {len(docs)} ({kb.get('chunk_count', 0)} chunks)")
                for d in docs:
                    st.caption(f"📄 {d}")
            else:
                st.warning("No chapters loaded yet. Upload a PDF below.")
        except requests.exceptions.RequestException:
            st.caption("(Couldn't load chapter list — backend is busy.)")
    st.divider()

    # PDF upload: lets a student add a new chapter without touching the code.
    # The file is sent to the backend's /upload endpoint, which ingests it into
    # Qdrant in the background.
    st.markdown("**Add a Chapter (PDF)**")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("Add to Knowledge Base"):
            with st.spinner("Uploading and processing..."):
                try:
                    resp = requests.post(
                        UPLOAD_URL,
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    )
                    resp.raise_for_status()
                    st.success("Uploaded! Processing in the background — give it a minute, then ask away.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Upload failed: {e}")
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Danger zone: wipes every chapter from the vector DB. Two-step confirm so a
    # kid can't nuke the knowledge base with one accidental click.
    st.divider()
    st.markdown("**Danger Zone**")
    if st.checkbox("I want to delete the knowledge base"):
        if st.button("Delete Knowledge Base", type="primary"):
            with st.spinner("Deleting..."):
                try:
                    resp = requests.delete(KB_URL)
                    resp.raise_for_status()
                    st.warning(resp.json().get("message", "Knowledge base deleted."))
                except requests.exceptions.RequestException as e:
                    st.error(f"Delete failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Tutor. Ask me to **explain a concept** or **give you a quiz**.", "type": "text"}]

st.title("AI Science Tutor")
st.caption("Powered by Decoupled Agentic RAG")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "quiz":
            st.write("**Quiz Generated**")
            for idx, q in enumerate(msg["content"]):
                with st.expander(f"Q{idx+1}: {q['question']}", expanded=True):
                    st.radio("Options:", q['options'], key=f"hist_q_{idx}_{len(st.session_state.messages)}")
                    st.success(f"Correct Answer: {q['answer']}")
                    st.info(f"Explanation: {q['explanation']}")
        else:
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **Page {s['page']}** ({s['topic']}): _{s['preview']}_")

# Handle new user input
if prompt := st.chat_input("Ask about chemical reactions, equations, or request a quiz..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing query via API..."):
            try:
                # Send request to FastAPI backend. Timeout so the UI fails with a
                # clear message instead of spinning forever if the backend hangs.
                response = requests.post(
                    API_URL,
                    json={"query": prompt, "history": [[m["role"], str(m["content"])] for m in st.session_state.messages[-5:]]},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                
                intent = data.get("intent", "CHAT")
                response_content = data.get("response", "")
                sources = data.get("sources", [])

                # Render a quiz only when the response is actually a list of
                # questions. A QUIZ intent can still return a plain string — e.g.
                # an out-of-scope query the backend refuses, or a quiz-agent
                # error fallback — and iterating that string would crash with
                # "string indices must be integers".
                if "QUIZ" in intent and isinstance(response_content, list) and response_content:
                    st.write("**Quiz generated based on your request.**")
                    for idx, q in enumerate(response_content):
                        with st.expander(f"Q{idx+1}: {q['question']}", expanded=True):
                            st.radio("Select an option:", q['options'], key=f"live_q_{idx}")
                            st.caption("*(Answer revealed in history)*")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "type": "quiz",
                        "sources": sources
                    })
                else:
                    # Coerce non-string payloads (e.g. an empty quiz list from a
                    # generation error) into a readable message so st.markdown
                    # never receives a list.
                    text = response_content if isinstance(response_content, str) else \
                        "Sorry, I couldn't generate a response for that. Try a topic from the loaded chapters."
                    st.markdown(text)
                    if sources:
                        with st.expander("Sources Used"):
                            for s in sources:
                                st.markdown(f"- **Page {s['page']}** ({s['topic']})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": text,
                        "type": "text",
                        "sources": sources
                    })
            except requests.exceptions.RequestException as e:
                st.error("Backend is unreachable. Ensure FastAPI is running on 127.0.0.1:8000.")