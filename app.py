import os
import streamlit as st
import requests

st.set_page_config(page_title="Sage | Adaptive RAG Tutor", page_icon="🦉", layout="wide")

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
def _resolve_backend():
    # 1. Env var
    env_url = os.environ.get("BACKEND_URL")
    if env_url:
        return env_url
    # 2. Streamlit Cloud secret. Accessing st.secrets when no secrets file
    #    exists (e.g. local dev) raises, so we catch and fall through to the
    #    local default. On Streamlit Cloud the dashboard secret is injected and
    #    this returns the configured backend URL.
    try:
        secret_url = st.secrets["BACKEND_URL"]
        if secret_url:
            return secret_url
    except Exception:
        pass
    # 3. Local fallback
    return "http://127.0.0.1:8000"

BACKEND = _resolve_backend()
API_URL = f"{BACKEND}/query"
UPLOAD_URL = f"{BACKEND}/upload"
KB_URL = f"{BACKEND}/knowledge-base"
HEALTH_URL = f"{BACKEND}/health"

with st.sidebar:
    st.title("🦉 Sage · Settings")
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
                st.success(f"Notes loaded: {len(docs)} ({kb.get('chunk_count', 0)} chunks)")
                for d in docs:
                    st.caption(f"📄 {d}")
            else:
                st.warning("No notes loaded yet. Upload a PDF below.")
        except requests.exceptions.RequestException:
            st.caption("(Couldn't load chapter list — backend is busy.)")
    st.divider()

    # PDF upload: lets a student add a new chapter without touching the code.
    # The file is sent to the backend's /upload endpoint, which ingests it into
    # Qdrant in the background.
    st.markdown("**Add Notes (PDF)**")
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
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm **Sage** 🦉 — upload your notes, then ask me to **explain a concept** or **quiz you**.", "type": "text"}]

st.title("🦉 Sage: Adaptive RAG Tutor")
st.caption("Learn from your own notes · powered by Adaptive RAG")

def render_quiz(quiz, qid):
    """
    Interactive MCQ quiz. Options start UNSELECTED (index=None) and the answers
    are revealed only after the user submits — so it behaves like a real quiz
    instead of pre-selecting "A" and leaking the answer on the first click.
    Widgets live inside an st.form, so picking options does NOT trigger reruns;
    only "Submit Answers" does. `qid` (the message's stable list index) keeps the
    widget keys unique and stable across reruns.
    """
    done_key = f"quiz_done_{qid}"
    done = st.session_state.get(done_key, False)

    with st.form(key=f"quizform_{qid}"):
        st.write("**Quiz** — pick an answer for each question, then submit.")
        for idx, q in enumerate(quiz):
            st.radio(
                f"**Q{idx + 1}: {q['question']}**",
                q["options"],
                index=None,
                key=f"quiz_{qid}_{idx}",
                disabled=done,
            )
        submitted = st.form_submit_button("Submit Answers", disabled=done)

    if submitted:
        st.session_state[done_key] = True
        done = True

    if done:
        score = 0
        for idx, q in enumerate(quiz):
            chosen = st.session_state.get(f"quiz_{qid}_{idx}")
            answer_letter = str(q.get("answer", "")).strip().upper()
            # Options look like "C. It forms white powder"; the answer is just "C".
            correct_option = next(
                (o for o in q["options"] if o.strip().upper().startswith(answer_letter)),
                q.get("answer", ""),
            )
            if chosen is not None and chosen == correct_option:
                score += 1
                st.success(f"Q{idx + 1}: Correct ✅  ({correct_option})")
            else:
                st.error(f"Q{idx + 1}: Your answer — {chosen if chosen else 'none selected'}")
                st.info(f"Correct answer: {correct_option}")
            st.caption(f"Explanation: {q['explanation']}")
        st.markdown(f"### Score: {score} / {len(quiz)}")


# Render chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "quiz":
            render_quiz(msg["content"], qid=i)
        else:
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **Page {s['page']}** ({s['topic']}): _{s['preview']}_")

# Handle new user input
if prompt := st.chat_input("Ask a question about your notes, or say 'quiz me'..."):
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
                    # Store the quiz, then rerun so the history loop renders it
                    # through the single interactive path (render_quiz: a form
                    # with unselected options + a Submit button). Rendering it
                    # inline here too would create a second, divergent copy of
                    # the quiz widgets that leaks the answer on first click.
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "type": "quiz",
                        "sources": sources
                    })
                    st.rerun()
                else:
                    # Coerce non-string payloads (e.g. an empty quiz list from a
                    # generation error) into a readable message so st.markdown
                    # never receives a list.
                    text = response_content if isinstance(response_content, str) else \
                        "Sorry, I couldn't generate a response for that. Try a topic from your loaded notes."
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