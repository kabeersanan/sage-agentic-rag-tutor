import asyncio
import numpy as np

from src.database.vector_store import get_embedding_function


class SemanticCache:
    """
    Caches answers keyed by query *meaning* (embedding) rather than exact text.

    A new query is served from cache when its cosine similarity to a previously
    seen query exceeds `threshold`. This means paraphrases of the same question
    ("Explain displacement reactions" vs "what is a displacement reaction?") all
    hit a single cached answer, bypassing ChromaDB retrieval and the Groq LLM.

    Eviction is FIFO (oldest entry dropped first) once `maxsize` is exceeded.
    A linear cosine scan over a small cache (~100 entries) is microseconds; a
    real vector index (e.g. FAISS) would only matter at thousands of entries.
    """

    def __init__(self, threshold: float = 0.95, maxsize: int = 100):
        self.threshold = threshold
        self.maxsize = maxsize
        # Reuse the SAME mpnet embedder the vector store uses — no extra model load.
        self._embedder = get_embedding_function()
        self._vectors = []   # list[np.ndarray]: normalized cached query embeddings
        self._values = []    # parallel list: cached QueryResponse objects

    def _embed(self, text: str) -> np.ndarray:
        """Embed and L2-normalize so a dot product equals cosine similarity."""
        vec = np.array(self._embedder.embed_query(text), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    async def get(self, query: str):
        """Return a cached value for a semantically-similar query, or None."""
        if not self._vectors:
            return None
        # Embedding is synchronous CPU work (~30-60ms); offload it so we don't
        # block the event loop and stall other concurrent requests.
        q = await asyncio.to_thread(self._embed, query)
        sims = np.dot(np.vstack(self._vectors), q)   # cosine vs all cached at once
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return self._values[best]                # SEMANTIC HIT
        return None

    async def set(self, query: str, value) -> None:
        """Store a query/answer pair, evicting the oldest entry if full."""
        vec = await asyncio.to_thread(self._embed, query)
        self._vectors.append(vec)
        self._values.append(value)
        if len(self._vectors) > self.maxsize:
            self._vectors.pop(0)
            self._values.pop(0)

    def clear(self) -> None:
        self._vectors.clear()
        self._values.clear()


# Single shared instance for the app.
query_cache = SemanticCache(threshold=0.95, maxsize=100)
