import os
import re
# Using Local Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_COLLECTION_NAME as COLLECTION_NAME

# The mpnet model is ~420MB and takes several seconds to load. Loading it on
# every call (ingest, query, cache) was the main source of latency, so we load
# it ONCE and reuse the instance.
_embedding_fn = None

def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        # Using Local MPNet (512 tokens) as discussed
        _embedding_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
    return _embedding_fn

# Reuse a single Qdrant client so the TLS connection stays warm (HTTP
# keep-alive) instead of paying a fresh handshake + version-check round trip on
# every call. check_compatibility=False also silences the version-mismatch noise.
_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            check_compatibility=False,
        )
    return _qdrant_client

def create_vector_db(chunks):
    if not chunks: return
    
    embedding_fn = get_embedding_function()
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    # Reuse the shared client to ensure the collection exists before uploading
    client = get_qdrant_client()

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # 768 is the exact output dimension for all-mpnet-base-v2
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        # Index metadata.source so we can later filter/delete by source file
        # (needed for idempotent re-uploads). Qdrant rejects such filters
        # without an index on the field.
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.source",
            field_schema="keyword",
        )

    print(f"Uploading {len(chunks)} chunks to Qdrant Cloud...")
    
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        url=url,
        api_key=api_key,
        collection_name=COLLECTION_NAME,
    )

def delete_by_source(source_path):
    """
    Removes all chunks that came from a given source file. Called before
    re-ingesting an uploaded file so that re-uploading the same PDF REPLACES
    its old vectors instead of appending duplicate copies.
    Safe to call when the collection or those points don't exist.
    """
    from qdrant_client.http import models
    client = get_qdrant_client()
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return
    # Qdrant can only filter/delete by a payload field if it's indexed. Ensure a
    # keyword index on metadata.source exists (no-op if already created).
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.source",
            field_schema="keyword",
        )
    except Exception:
        pass  # already exists
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=source_path),
            )]
        ),
    )

def get_knowledge_base_stats():
    """
    Returns a summary of what's currently in the Qdrant collection:
        {"chunk_count": int, "documents": [filename, ...]}
    `documents` is the list of distinct source PDFs (basenames only).
    Returns zeros/empty if the collection doesn't exist yet.
    """
    client = get_qdrant_client()
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return {"chunk_count": 0, "documents": []}

    chunk_count = client.count(collection_name=COLLECTION_NAME).count

    # Scroll through payloads to collect distinct source files. langchain_qdrant
    # nests the doc metadata under the "metadata" payload key.
    sources = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["metadata"],
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for p in points:
            src = (p.payload or {}).get("metadata", {}).get("source")
            if src:
                # Strip BOTH path separators: files ingested on Windows store
                # backslash paths, but os.path.basename on Render's Linux won't
                # split on "\", leaving an ugly "data\raw\file.pdf".
                sources.add(re.split(r"[\\/]", src)[-1])
        if offset is None:   # no more pages
            break

    return {"chunk_count": chunk_count, "documents": sorted(sources)}

def delete_vector_db():
    """
    Deletes the entire Qdrant collection, wiping the knowledge base.
    Safe to call even if the collection doesn't exist.
    Returns True if a collection was deleted, False if there was nothing to delete.
    """
    client = get_qdrant_client()
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return False
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Deleted Qdrant collection '{COLLECTION_NAME}'.")
    return True