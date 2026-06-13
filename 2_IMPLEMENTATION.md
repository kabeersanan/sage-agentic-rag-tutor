# 2. Feature-by-Feature Implementation Walkthrough

> Goal: for **every** feature, know the exact file, the exact function, and how the pieces talk to each other. Each feature has: *what it does → how it works step-by-step → the key code → the "gotcha" you should mention.*

Features covered:
1. PDF Ingestion (loading + chunking)
2. Embeddings & the Vector Store (writing to Qdrant)
3. Idempotent single-file upload (no duplicates)
4. Retrieval with a relevance threshold (the anti-hallucination guard)
5. The Agentic Router (intent classification)
6. The Concept Agent (explanations + memory)
7. The Quiz Agent (structured JSON generation)
8. The Semantic Cache (skip repeated work)
9. Knowledge-base management (stats / delete)
10. Performance: singletons for the model and the DB client
11. App lifecycle: lifespan, CORS, background tasks
12. The Streamlit frontend (state, sidebar, rendering)

---

## Feature 1 — PDF Ingestion: loading and chunking

**What it does:** Turns a raw PDF into many small, clean text pieces ("chunks") with metadata, ready to be embedded.

### Step 1a: Load the PDF → `src/ingestion/pdf_loader.py`
There are two functions:
- `load_documents()` — loads **every** PDF in `data/raw/` (used for full rebuilds / the CLI).
- `load_single_document(file_path)` — loads **one** PDF (used by the upload endpoint, so we don't re-process old files).

Both use LangChain's `PyPDFLoader`, which returns **one `Document` object per page**, each carrying metadata like the page number and source filename.

```python
def load_single_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()        # one Document per page
    return docs
```

**Gotcha to mention:** loading per-page matters because the page number becomes the **citation** the student sees later ("Page 25").

### Step 1b: Split into chunks → `src/ingestion/chunker.py`
**Why chunk at all?** A whole page is too big to embed meaningfully and too big to feed the LLM efficiently. We cut each page into ~400-token pieces.

The smart part is **how** we cut. We use `RecursiveCharacterTextSplitter` with a **priority list of separators**:

```python
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=400,        # ~400 tokens per chunk
    chunk_overlap=50,      # 50 tokens repeated between neighbors
    separators=["\n\n", "\n", ". ", "? ", " ", ""]  # try these in order
)
```

**How the separator priority works (explain it like this):** The splitter tries to break on **paragraphs (`\n\n`) first**. Only if a paragraph is still too big does it fall back to breaking on **lines (`\n`)**, then **sentences (`. `)**, and so on down to single characters as a last resort. This keeps related ideas together instead of blindly cutting at character 400 (which could slice a sentence in half).

> Analogy: When tearing a sheet of paper, you'd tear along the dotted lines first, not randomly through the middle of words.

**Chunk overlap (50 tokens):** each chunk repeats the last ~50 tokens of the previous one. This prevents losing meaning at the boundary — if an important sentence sits right on the cut line, it appears in **both** chunks so retrieval can still find it.

**Metadata enrichment:** for each chunk we attach:
- `page` (for citations),
- `topic` — extracted by `extract_topic_header()`, a heuristic that grabs the first short, title-like line of the chunk,
- `token_count` — for validation.

```python
meta['topic'] = extract_topic_header(chunk.page_content)
meta['token_count'] = count_tokens(chunk.page_content)
```

**Key files working together:** `pdf_loader.py` (raw text) → `chunker.py` (small enriched chunks). Both are pure functions with no side effects, which is why they're the easiest to unit-test (`tests/test_chunker.py`).

---

## Feature 2 — Embeddings & writing to the Vector Store → `src/database/vector_store.py`

**What it does:** Converts each chunk's text into a 768-number vector and uploads it to Qdrant Cloud.

### The embedding function (and the singleton trick)
```python
_embedding_fn = None   # module-level cache

def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
    return _embedding_fn
```
**Why the `if _embedding_fn is None` check?** The model takes **~12 seconds to load**. Without this cache, every query and every upload would reload it. With it, the model loads **once per process** and is reused everywhere. (This was a real performance fix — see Feature 10.)

### Writing chunks to Qdrant — `create_vector_db(chunks)`
Step by step:
1. Get the (cached) embedding function.
2. Get the (cached) Qdrant client.
3. If the collection doesn't exist yet, **create it** with vector size **768** and **Cosine** distance, and **create a payload index on `metadata.source`** (needed later for delete-by-file).
4. Call `QdrantVectorStore.from_documents(...)` which **embeds every chunk and uploads** them in one go.

```python
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    client.create_payload_index(            # so we can filter/delete by source file later
        collection_name=COLLECTION_NAME,
        field_name="metadata.source",
        field_schema="keyword",
    )

QdrantVectorStore.from_documents(
    documents=chunks, embedding=embedding_fn,
    url=url, api_key=api_key, collection_name=COLLECTION_NAME,
)
```

**Two numbers you must know:**
- **768** = the output dimension of `all-mpnet-base-v2`. The collection's vector size must match exactly or Qdrant rejects the upload.
- **Cosine** = the "distance" metric. Cosine similarity measures the *angle* between two vectors — great for "are these two texts about the same thing?" regardless of length.

---

## Feature 3 — Idempotent single-file upload (no duplicates)

**The problem it solves:** If a student uploads `Trees.pdf` twice, a naive system stores its chunks **twice** → duplicate, polluted results. ("Idempotent" = doing it twice has the same effect as doing it once.)

**How it's solved** (`ingest_file` in `server.py` + `delete_by_source` in `vector_store.py`):
Before inserting a file's chunks, we **delete any existing chunks from that same source file**:

```python
# in ingest_file() background task:
delete_by_source(file_path)   # remove old copies of THIS file
create_vector_db(chunks)      # then insert the fresh copies
```

```python
def delete_by_source(source_path):
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME): return
    client.create_payload_index(...)   # ensure index exists (no-op if already there)
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(must=[models.FieldCondition(
            key="metadata.source", match=models.MatchValue(value=source_path))]),
    )
```

**The big gotcha (a perfect "tell me about a bug you fixed" story):** Qdrant **refuses to filter or delete by a field unless that field is indexed.** The first attempt failed with `400 Bad Request: Index required for "metadata.source"`. The fix was to **create a `keyword` payload index** on `metadata.source` — both when the collection is created and defensively inside `delete_by_source`. Now re-uploading a file **replaces** its chunks instead of duplicating them.

---

## Feature 4 — Retrieval with a relevance threshold (the anti-hallucination guard) → `src/database/retriever.py`

**What it does:** Given a question, find the most relevant chunks — **but only if they're actually relevant.** If nothing is relevant, return **nothing**, so the system can honestly refuse instead of making things up.

### The core function
```python
async def aretrieve_relevant(vector_store, query, k=4, threshold=SIMILARITY_THRESHOLD):
    pairs = await vector_store.asimilarity_search_with_score(query, k=k)
    return [doc for doc, score in pairs if score >= threshold]   # keep only good matches
```

Step by step:
1. The query text is embedded into a vector (same model as the chunks).
2. Qdrant returns the `k=4` nearest chunks **with their raw cosine scores** (0 to 1, higher = more similar).
3. We **drop any chunk whose score is below `SIMILARITY_THRESHOLD = 0.3`** (`src/config.py`).
4. If all 4 are below threshold, the function returns `[]` (empty).

### Why a threshold at all (the real bug story)
Without it, the retriever **always** returns its 4 nearest chunks — even for a totally unrelated question. Example you actually hit: with only graph-theory notes loaded, asking *"explain acid, bases and salts"* returned graph chunks with similarity ~0.09 (basically noise), and the LLM **confidently explained graphs**. That's a hallucination caused by garbage context.

Calibration from real data (memorize these numbers — they prove you measured, not guessed):

| Query | Top cosine score | In the notes? |
|---|---|---|
| "BFS and DFS" | 0.738 | ✅ |
| "Dijkstra shortest path" | 0.616 | ✅ |
| "strongly connected components" | 0.32 | ✅ (borderline) |
| "photosynthesis" | 0.195 | ❌ |
| "acid bases and salts" | 0.106 | ❌ |

A threshold of **0.3** cleanly separates real matches (≥0.32) from junk (≤0.2).

### The "don't trust the library" gotcha (a strong senior-level point)
LangChain's built-in `score_threshold` option **didn't work** because it **re-normalizes** the raw cosine score with an opaque formula (it mapped a raw 0.106 up to ~0.55, so the threshold let junk through). The fix was to **filter on the raw Qdrant cosine score myself** using `asimilarity_search_with_score`, rather than trusting the framework's hidden mapping. Say: *"I verified the actual scores, found the framework's normalization was untrustworthy for a hard cutoff, and did the filtering explicitly on raw scores."*

### What happens when retrieval returns empty → `server.py`
```python
if "CHAT" not in intent and not retrieved_docs:
    msg = await asyncio.to_thread(build_no_context_message)
    return QueryResponse(intent=intent, response=msg, sources=[])
```
The tutor returns a refusal that **lists the currently loaded chapters** so the student knows what they *can* ask about. Greetings (`CHAT`) are exempt — "hello" doesn't need context.

---

## Feature 5 — The Agentic Router → `src/agents/router.py` + `src/agents/prompts.py`

**What it does:** Reads the user's question and outputs exactly **one word**: `QUIZ`, `EXPLAIN`, or `CHAT`.

```python
async def route_query(query):
    llm = ChatGroq(model=LLM_MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.0)
    prompt = ChatPromptTemplate.from_template(ROUTER_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()
    intent = await chain.ainvoke({"query": query})
    return intent.strip().upper()
```

How it works:
- The `ROUTER_SYSTEM_PROMPT` instructs the model: *"Return ONLY one word: QUIZ / EXPLAIN / CHAT."*
- `temperature=0.0` → **no randomness**. Classification must be consistent; the same question should always route the same way.
- `prompt | llm | StrOutputParser()` is a LangChain **chain**: format the prompt → send to Groq → extract plain text.
- `.strip().upper()` normalizes the output so downstream `if "QUIZ" in intent` checks are reliable even if the model adds spaces/casing.

**Gotcha to mention:** the downstream code uses substring checks (`"QUIZ" in intent`) rather than exact equality, which is a small robustness trick in case the model returns `"QUIZ."` or `"Quiz"`.

---

## Feature 6 — The Concept Agent (explanations + memory) → `src/agents/concept_agent.py`

**What it does:** Writes a friendly, analogy-driven explanation grounded **only** in the retrieved context, while remembering recent conversation so follow-ups with pronouns work.

```python
async def generate_explanation(query, context, history):
    llm = ChatGroq(model=LLM_MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(CONCEPT_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()
    history_str = "\n".join([f"{role}: {msg}" for role, msg in history])
    return await chain.ainvoke({"query": query, "context": context, "history": history_str})
```

Key points:
- **`temperature=0.3`** — a little creativity so explanations feel human, but still grounded.
- **The prompt enforces grounding:** `CONCEPT_SYSTEM_PROMPT` says *"Use ONLY the provided Context. If the answer is not there, say: 'I cannot find this specific detail in the notes provided.'"* This is the **second layer** of anti-hallucination defense (the first being the retrieval threshold).
- **Memory:** the `history` is a list of `(role, message)` pairs. We flatten it into a string and inject it into the prompt so the model can resolve "he/it/that."

**How memory actually flows (important detail):** the conversation memory is **held by the frontend**, not the backend. Streamlit keeps the full chat in `st.session_state.messages` and sends the **last 5 turns** with each request. The backend is **stateless** — it doesn't remember anything between calls. (This is a deliberate design choice; see file 4 for why stateless backends scale better.)

---

## Feature 7 — The Quiz Agent (structured JSON) → `src/agents/quiz_agent.py`

**What it does:** Generates exactly 3 multiple-choice questions as a **valid JSON array** the frontend can render.

The challenge: LLMs love to wrap JSON in markdown fences (```` ```json ... ``` ````) or add chatty text, which breaks `json.loads()`. Two defenses:

**Defense 1 — low temperature:** `temperature=0.1` for strict, predictable formatting.

**Defense 2 — a cleaning function:**
```python
def clean_json_text(text):
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()   # pull JSON out of the code fence
    return text
```
Then:
```python
content = raw_response.content
cleaned = clean_json_text(content)
quiz_data = json.loads(cleaned)     # parse to a Python list of dicts
```

**Defense 3 — graceful failure:** if `json.loads` still fails, we **don't crash**. We return a friendly fallback question explaining the error, so the UI never breaks:
```python
except json.JSONDecodeError:
    return [{"question": "Error generating quiz questions.", "options": [...],
             "answer": "A", "explanation": "The AI returned invalid JSON format."}]
```

**Gotcha to mention:** `clean_json_text` is the most-tested function in the repo (`tests/test_agents.py`) precisely because flaky LLM JSON is the most fragile part of the system.

---

## Feature 8 — The Semantic Cache → `src/api/cache.py`

**What it does:** Remembers past answers so that asking the *same thing in different words* returns instantly, skipping both the vector search and the LLM call.

**Normal cache vs. semantic cache (explain the difference):**
- A normal cache matches **exact text**. "What is a chemical reaction?" and "what's a chemical reaction" would be two different keys → two misses.
- A **semantic cache** matches **meaning**. It embeds the query and compares it to past queries by **cosine similarity**. If similarity ≥ **0.95**, it's "the same question" → return the stored answer.

```python
async def get(self, query):
    if not self._vectors: return None
    q = await asyncio.to_thread(self._embed, query)   # embed off the event loop
    sims = np.dot(np.vstack(self._vectors), q)        # cosine vs ALL cached at once
    best = int(np.argmax(sims))
    if sims[best] >= self.threshold:                  # 0.95
        return self._values[best]                     # CACHE HIT
    return None
```

How it works step by step:
1. Embed the incoming query and **L2-normalize** it (so a dot product equals cosine similarity).
2. `np.dot(all_cached_vectors, query)` computes similarity against **every** cached query at once (fast vectorized math).
3. Take the best match; if it clears 0.95, return the stored `QueryResponse`.
4. Eviction is **FIFO** — once 100 entries are stored, the oldest is dropped.

**Important design choices (and gotchas):**
- It **reuses the same embedding model** as the vector store (no second model load).
- Embedding is offloaded with `asyncio.to_thread` so the ~30–60ms CPU work doesn't freeze the server's event loop.
- **Only stateless queries are cached** — in `server.py`, `use_cache = not request.history`. A follow-up like "when was *he* born?" depends on previous turns, so caching it by the query text alone would be wrong.
- The cache is **cleared whenever the knowledge base changes** (on upload and on delete), because old answers may now be stale.
- **Limitation to admit:** the cache is **in-memory and per-process**. If you run multiple backend copies, each has its own cache, and a restart wipes it. The fix at scale is a shared cache like Redis (file 4).

---

## Feature 9 — Knowledge-base management → `server.py` + `vector_store.py`

Three endpoints manage the "library":
- **`GET /knowledge-base`** → `get_knowledge_base_stats()` scrolls the collection and returns `{chunk_count, documents: [...]}`. Powers the sidebar "Chapters loaded" indicator and the refusal message.
- **`DELETE /knowledge-base`** → `delete_vector_db()` drops the whole Qdrant collection, sets `_vector_store = None`, and clears the cache.
- **`POST /upload`** → saves the file and triggers background ingestion (Feature 3).

```python
def get_knowledge_base_stats():
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"chunk_count": 0, "documents": []}
    chunk_count = client.count(COLLECTION_NAME).count
    sources = set()
    offset = None
    while True:                                  # paginate through all points
        points, offset = client.scroll(COLLECTION_NAME, with_payload=["metadata"],
                                       with_vectors=False, limit=256, offset=offset)
        for p in points:
            src = p.payload.get("metadata", {}).get("source")
            if src: sources.add(os.path.basename(src))
        if offset is None: break
    return {"chunk_count": chunk_count, "documents": sorted(sources)}
```
**Gotcha:** we ask Qdrant for `with_vectors=False` — we only need the metadata, not the heavy 768-number vectors, so the call stays light.

---

## Feature 10 — Performance: singletons for the model and the DB client

This is a **headline optimization story.** Originally, every operation created a **new** embedding model (12.6s load) and a **new** Qdrant client (which does a slow server "compatibility check" round-trip). That made queries and uploads painfully slow, and a slow stats call even made the UI falsely show "backend offline."

The fix — two module-level singletons in `vector_store.py`:
```python
_embedding_fn = None        # load the 420MB model ONCE
_qdrant_client = None        # reuse ONE warm connection (keep-alive)

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=..., api_key=...,
                                      check_compatibility=False)  # skip slow version round-trip
    return _qdrant_client
```

**Measured results (quote these):**
- Embedding model: **12.6s → loaded once, then 0s.**
- Qdrant client init: **2.6s → 0.79s** (0s when reused).
- `/knowledge-base` stats call: **4.77s → ~1s.**

`check_compatibility=False` does double duty: it skips a network round-trip **and** silences the noisy "client/server version mismatch" warnings.

---

## Feature 11 — App lifecycle: lifespan, CORS, background tasks → `server.py`

**Lifespan (startup/shutdown hook):**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vector_store
    try:
        _vector_store = get_vector_store()      # connect to Qdrant at boot
        print("Connected to Qdrant Cloud collection.")
    except Exception as e:
        _vector_store = None                    # don't crash; /health will report it
        print(f"WARNING: ...{e}")
    yield                                        # server runs here
```
**Gotcha (real bug fixed):** the connection is wrapped in `try/except` so a Qdrant hiccup **doesn't crash uvicorn at startup**. Earlier, an unhandled error here meant the server never bound port 8000 and the whole app looked dead. Now it boots and `/health` honestly reports the problem.

**CORS:** `app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)` lets the browser-based Streamlit app call the API. (For production you'd lock `allow_origins` to your real domain.)

**Background tasks:** `/upload` returns **immediately** and runs `ingest_file` via FastAPI's `BackgroundTasks`. The student isn't forced to stare at a spinner for 40 seconds while chunks embed.
```python
background_tasks.add_task(ingest_file, file_path)
return {"status": "success", "message": "...Ingestion started in background."}
```
**Gotcha to admit:** `BackgroundTasks` runs **inside the same server process**. Heavy CPU embedding can still slow other requests (GIL contention). The real-world fix is an external task queue (Celery/RQ) — see file 4.

---

## Feature 12 — The Streamlit frontend → `app.py`

**State management:** Streamlit re-runs the whole script on every interaction, so persistent data lives in `st.session_state.messages` (the full chat history). On a new message:
1. Append the user's message to `session_state`.
2. POST to the backend, sending the **last 5 turns** as history.
3. Render the response; if it's a quiz, render expandable question cards; else render markdown + a "Sources" expander.
4. Append the assistant's reply to `session_state`.

```python
response = requests.post(API_URL,
    json={"query": prompt,
          "history": [[m["role"], str(m["content"])] for m in st.session_state.messages[-5:]]},
    timeout=60)
```

**The sidebar** shows live system status, and this is where two real bugs were fixed:
- **Connectivity uses `/health` (fast ~0.2s), not `/knowledge-base` (slow ~1–5s)**, so a slow stats call can't make a healthy backend look "offline." The chapter list loads separately with a longer timeout and a graceful fallback.
- **URLs use `http://127.0.0.1:8000`, not `localhost`.** On Windows, `localhost` can resolve to IPv6 `::1` while uvicorn binds IPv4 `127.0.0.1`, causing "backend unreachable" even when it's running. Using the explicit IPv4 address fixes it.

```python
BACKEND = "http://127.0.0.1:8000"      # NOT "localhost" — avoids Windows IPv6 trap
# ...
try:
    requests.get(HEALTH_URL, timeout=5).raise_for_status()   # fast connectivity check
    st.text("Backend: Connected")
except requests.exceptions.RequestException:
    st.error("Backend offline — start it on 127.0.0.1:8000.")
```

Continue to [3_DATA_FLOW.md](3_DATA_FLOW.md).
