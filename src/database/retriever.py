import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from src.database.vector_store import get_embedding_function
from src.config import QDRANT_COLLECTION_NAME as COLLECTION_NAME
from src.config import SIMILARITY_THRESHOLD

def get_vector_store():
    """
    Returns a QdrantVectorStore bound to the cloud collection. Use its
    similarity_search_with_score() to get RAW cosine scores, which we filter
    with SIMILARITY_THRESHOLD ourselves — langchain's built-in score_threshold
    re-normalizes scores opaquely and can't be trusted for a hard cutoff.
    """
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embedding_function(),
        collection_name=COLLECTION_NAME,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

async def aretrieve_relevant(vector_store, query, k=4, threshold=SIMILARITY_THRESHOLD):
    """
    Retrieve up to k chunks whose RAW cosine similarity >= threshold.
    Returns [] when nothing is relevant, so callers can refuse to answer
    instead of feeding the LLM near-random context.
    """
    pairs = await vector_store.asimilarity_search_with_score(query, k=k)
    return [doc for doc, score in pairs if score >= threshold]

def get_retriever(k=4, student_id=None):
    embedding_fn = get_embedding_function()

    # Connect to the existing cloud collection
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    search_kwargs = {"k": k}

    # Apply metadata filtering if a student_id is provided
    if student_id:
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.student_id",
                    match=models.MatchValue(value=student_id),
                )
            ]
        )

    return vector_store.as_retriever(search_kwargs=search_kwargs)