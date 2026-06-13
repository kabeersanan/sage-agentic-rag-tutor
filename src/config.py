import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- API KEYS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- FILE PATHS ---
# We use os.path.join to make sure it works on both Windows and Mac
DATA_DIR = os.path.join("data", "raw")
DB_DIR = os.path.join("data", "vector_store")

# --- QDRANT CLOUD ---
# Single source of truth for the collection name. Defined here so the writer
# (vector_store) and reader (retriever) can never drift to different names.
QDRANT_COLLECTION_NAME = "sage_agentic_rag_tutor"

# --- RAG SETTINGS ---
# Minimum cosine relevance (0-1) a retrieved chunk must have to be used as
# context. Below this, the query is treated as "not in the notes" so the tutor
# refuses instead of hallucinating from near-zero-similarity chunks.
# Calibrated: real matches score >0.3, off-topic queries score <0.2.
SIMILARITY_THRESHOLD = 0.3
# Chunk Size 1000: Good balance. Large enough to capture full context (approx 2-3 paragraphs).
# Overlap 200: Ensures we don't cut a sentence in half at the edge of a chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- AI MODELS ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Using Gemini Flash because it is fast, cheap, and has a large context window
LLM_MODEL_NAME = "llama-3.3-70b-versatile"