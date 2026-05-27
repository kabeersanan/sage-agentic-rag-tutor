import streamlit as st
import time
import os

from src.database.retriever import get_retriever
from src.agents.router import route_query
from src.agents.concept_agent import generate_explanation
from src.agents.quiz_agent import generate_quiz
from src.ingestion.pdf_loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.database.vector_store import create_vector_db
from src.config import DB_DIR

st.set_page_config(
    page_title="AI Tutor | Class 10 Science",
    layout="wide"
)

st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stButton button {border-radius: 20px;}
    .source-box {font-size: 0.8em; color: #666; background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_system():
    """
    Checks if the RAG system is ready. If not, runs ingestion.
    Cached to run once per session.
    """
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return True
    
    with st.status("Initializing system...", expanded=True) as status:
        st.write("Loading documents...")
        docs = load_documents()
        if not docs:
            st.error("No PDFs found in data/raw/. Please add a file.")
            st.stop()
            
        st.write("Chunking documents...")
        chunks = chunk_documents(docs)
        
        st.write("Building vector database...")
        create_vector_db(chunks)
        
        status.update(label="System ready.", state="complete", expanded=False)
    return True

@st.cache_resource
def load_retriever():
    """Load the Hybrid Retriever."""
    return get_retriever(k=4)

with st.sidebar:
    st.title("AI Tutor Settings")
    st.divider()
    
    st.markdown("**System Status**")
    st.text("Hybrid Search: Active")
    st.text("Metadata Filtering: Active")
    st.text("Knowledge Base: Class 10 Science Chapter 2")
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I am your AI Tutor. Ask me to **explain a concept** or **give you a quiz**.",
        "type": "text"
    })

initialize_system()
retriever = load_retriever()

st.title("AI Science Tutor")
st.caption("Powered by Hybrid RAG & Intelligent Agents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "quiz":
            st.write(f"**Quiz: {msg['topic']}**")
            for idx, q in enumerate(msg["content"]):
                with st.expander(f"Q{idx+1}: {q['question']}", expanded=True):
                    st.radio("Options:", q['options'], key=f"q_{idx}_{len(st.session_state.messages)}")
                    if st.button(f"Show Answer Q{idx+1}", key=f"btn_{idx}_{len(st.session_state.messages)}"):
                        st.success(f"Correct Answer: {q['answer']}")
                        st.info(f"Explanation: {q['explanation']}")
        else:
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("View Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **Page {s['page']}** ({s['topic']}): _{s['preview']}..._")

if prompt := st.chat_input("Ask about chemical reactions, equations, or request a quiz..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            
            retrieved_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in retrieved_docs])
            
            sources = [{
                "page": doc.metadata.get("page", "?"),
                "topic": doc.metadata.get("topic", "General"),
                "preview": doc.page_content[:100].replace("\n", " ")
            } for doc in retrieved_docs]

            intent = route_query(prompt).strip().upper()
            
            if "QUIZ" in intent:
                response = generate_quiz(prompt, context_text)
                st.write("**Quiz generated based on your request.**")
                
                for idx, q in enumerate(response):
                    with st.expander(f"Q{idx+1}: {q['question']}", expanded=True):
                        st.radio("Select an option:", q['options'], key=f"live_q_{idx}")
                        st.caption("*(Answer revealed in history)*")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "topic": prompt,
                    "type": "quiz",
                    "sources": sources
                })
                
            else:
                if "CHAT" in intent:
                    response = "I am here to help with Science! Try asking: 'Explain displacement reactions'."
                else:
                    response = generate_explanation(prompt, context_text, [])
                
                st.markdown(response)
                
                with st.expander("Sources Used"):
                    for s in sources:
                        st.markdown(f"- **Page {s['page']}** ({s['topic']})")

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "type": "text",
                    "sources": sources
                })