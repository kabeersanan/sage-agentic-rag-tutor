import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import QueryRequest, QueryResponse, HealthResponse, Source
from src.database.retriever import get_retriever
from src.agents.router import route_query
from src.agents.concept_agent import generate_explanation
from src.agents.quiz_agent import generate_quiz
from src.ingestion.pdf_loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.database.vector_store import create_vector_db
from src.config import DATA_DIR, DB_DIR

app = FastAPI(
    title="Sage AI Tutor API",
    description="Agentic RAG backend for Class 10 NCERT Science",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever lazily
_retriever = None

def load_system():
    global _retriever
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        _retriever = get_retriever(k=4)

@app.on_event("startup")
def startup_event():
    load_system()

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and the database is ready."""
    db_ready = os.path.exists(DB_DIR) and bool(os.listdir(DB_DIR))
    return HealthResponse(
        status="active",
        message="System is ready" if db_ready else "System active, but no Knowledge Base found."
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
        
    # Run ingestion in the background so the API doesn't hang
    background_tasks.add_task(rebuild_knowledge_base)
    
    return {"status": "success", "message": f"File {file.filename} uploaded. Ingestion started in background."}

def rebuild_knowledge_base():
    """Background task to process PDFs and build the vector DB."""
    global _retriever
    try:
        docs = load_documents()
        chunks = chunk_documents(docs)
        create_vector_db(chunks)
        _retriever = get_retriever(k=4) # Reload retriever
    except Exception as e:
        print(f"Error rebuilding knowledge base: {e}")

@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """Main endpoint to chat with the AI Tutor."""
    global _retriever
    if not _retriever:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized. Please upload a document first.")
    
    try:
        # 1. Retrieve Context
        retrieved_docs = _retriever.invoke(request.query)
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
        intent = route_query(request.query).strip().upper()
        
        # 4. Generate Response
        if "QUIZ" in intent:
            response_data = generate_quiz(request.query, context_text)
        elif "CHAT" in intent:
            response_data = "Hello! I am your AI Tutor. Ask me to explain a concept or give you a quiz!"
        else:
            # Format history for the agent
            history_tuples = [(msg[0], msg[1]) for msg in request.history] if request.history else []
            response_data = generate_explanation(request.query, context_text, history_tuples)

        return QueryResponse(
            intent=intent,
            response=response_data,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))