import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.schemas import QueryRequest, QueryResponse, HealthResponse, Source
from src.database.retriever import get_retriever, get_vector_store, aretrieve_relevant
from src.agents.router import route_query
from src.agents.concept_agent import generate_explanation
from src.agents.quiz_agent import generate_quiz
from src.ingestion.pdf_loader import load_documents, load_single_document
from src.ingestion.chunker import chunk_documents
from src.database.vector_store import create_vector_db, delete_vector_db, get_knowledge_base_stats, delete_by_source
from src.api.cache import query_cache
from src.config import DATA_DIR, DB_DIR

def build_no_context_message():
    """
    Refusal message for when a query doesn't match the loaded PDFs. Lists the
    currently loaded chapters so the student knows what they CAN ask about,
    instead of letting the LLM fabricate an answer from irrelevant chunks.
    """
    try:
        docs = get_knowledge_base_stats().get("documents", [])
    except Exception:
        docs = []
    base = (
        "I couldn't find anything about that in your uploaded notes. "
        "I can only answer from the PDFs currently loaded."
    )
    if docs:
        listed = "\n".join(f"- {d}" for d in docs)
        return (
            f"{base}\n\n**Currently loaded notes:**\n{listed}\n\n"
            "Try asking about one of these, or upload the relevant PDF."
        )
    return f"{base}\n\nNo notes are loaded yet — upload a PDF to get started."

# Vector store handle, initialized lazily (None = knowledge base not ready).
_vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic (Everything before the yield)
    print("Starting up Sage AI Tutor...")
    global _vector_store
    # Connect to the Qdrant Cloud collection. Wrapped in try/except so a
    # connection/collection error does NOT crash uvicorn startup — the server
    # still binds the port and /health can report that the KB is unavailable.
    try:
        _vector_store = get_vector_store()
        print("Connected to Qdrant Cloud collection.")
    except Exception as e:
        _vector_store = None
        print(f"WARNING: Could not connect to vector store on startup: {e}")

    yield # This is where FastAPI actually runs the server

    # Shutdown logic (Everything after the yield)
    print("Shutting down Sage AI Tutor...")
    # If you had database connections to close, you would do it here

# Single app definition with both the lifespan hook and CORS for the frontend.
app = FastAPI(
    title="Sage AI Tutor API",
    description="Agentic RAG backend for Class 10 NCERT Science",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and the vector store is connected."""
    # Knowledge base now lives in Qdrant Cloud, so readiness = retriever is
    # connected, not the presence of a local DB_DIR folder.
    db_ready = _vector_store is not None
    return HealthResponse(
        status="active",
        message="System is ready" if db_ready else "System active, but vector store is not connected."
    )

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and trigger background ingestion."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest ONLY this newly uploaded file in the background, so the API doesn't
    # hang and we don't re-process (and duplicate) already-ingested PDFs.
    background_tasks.add_task(ingest_file, file_path)

    return {"status": "success", "message": f"File {file.filename} uploaded. Ingestion started in background."}

def ingest_file(file_path: str):
    """Background task: process a single uploaded PDF and append it to the DB."""
    global _vector_store
    try:
        docs = load_single_document(file_path)
        chunks = chunk_documents(docs)
        # Idempotent re-upload: drop this file's previous chunks (if any) before
        # inserting, so uploading the same PDF twice replaces rather than dupes.
        delete_by_source(file_path)
        create_vector_db(chunks)          # appends to the existing collection
        _vector_store = get_vector_store() # (re)connect now that data exists
        # New docs change the knowledge base, so previously cached answers are
        # now stale — clear the cache to avoid serving outdated responses.
        query_cache.clear()
    except Exception as e:
        print(f"Error ingesting file {file_path}: {e}")

@app.get("/knowledge-base")
def knowledge_base_stats():
    """Report which documents (chapters) are currently loaded and chunk count."""
    try:
        return get_knowledge_base_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read knowledge base: {e}")

@app.delete("/knowledge-base")
def delete_knowledge_base():
    """Wipe the entire knowledge base (delete the Qdrant collection)."""
    global _vector_store
    try:
        deleted = delete_vector_db()
        _vector_store = None    # nothing to retrieve from anymore
        query_cache.clear()     # cached answers are now invalid
        msg = "Knowledge base deleted." if deleted else "Knowledge base was already empty."
        return {"status": "success", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge base: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Main endpoint to chat with the AI Tutor."""
    global _vector_store
    if not _vector_store:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized. Please upload a document first.")

    # 0. Semantic cache check.
    # Only cache stateless queries (no chat history). A query whose answer
    # depends on prior turns can't be safely keyed by the query alone.
    use_cache = not request.history
    if use_cache:
        cached = await query_cache.get(request.query)
        if cached is not None:
            return cached   # semantic hit: skips ChromaDB + Groq entirely

    try:
        # 1. Retrieve Context — only chunks above the relevance threshold, so an
        # off-topic query yields [] and we refuse instead of hallucinating.
        retrieved_docs = await aretrieve_relevant(_vector_store, request.query, k=4)
        context_text = "\n\n".join([d.page_content for d in retrieved_docs])

        # 2. Extract Sources
        sources = [
            Source(
                page=str(doc.metadata.get("page", "?")),
                topic=doc.metadata.get("topic", "General"),
                preview=doc.page_content[:100].replace("\n", " ") + "..."
            ) for doc in retrieved_docs
        ]

        # 3. Route Intent
        intent = (await route_query(request.query)).strip().upper()

        # No chunk cleared the relevance threshold => the topic isn't in the
        # loaded PDFs. Refuse (with the chapter list) rather than letting the LLM
        # hallucinate from unrelated context. Greetings (CHAT) still go through.
        # build_no_context_message hits Qdrant, so run it off the event loop.
        if "CHAT" not in intent and not retrieved_docs:
            msg = await asyncio.to_thread(build_no_context_message)
            return QueryResponse(intent=intent, response=msg, sources=[])

        # 4. Generate Response
        if "QUIZ" in intent:
            response_data = await generate_quiz(request.query, context_text)
        elif "CHAT" in intent:
            response_data = "Hello! I am your AI Tutor. Ask me to explain a concept or give you a quiz!"
        else:
            # Format history for the agent
            history_tuples = [(msg[0], msg[1]) for msg in request.history] if request.history else []
            response_data = await generate_explanation(request.query, context_text, history_tuples)

        result = QueryResponse(
            intent=intent,
            response=response_data,
            sources=sources
        )

        # 5. Store in cache for future semantically-similar queries.
        if use_cache:
            await query_cache.set(request.query, result)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))