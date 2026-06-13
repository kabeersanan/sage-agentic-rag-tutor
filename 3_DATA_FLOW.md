# 3. Deep-Dive Data Flow

> Goal: be able to **trace a request end-to-end out loud**, naming each file and function in order. Interviewers love "walk me through what happens when…" — these are your scripts.

---

## 3.1 Flow A — A student asks a question ("Explain photosynthesis")

### Text flowchart
```
[1] STUDENT types in Streamlit chat box                         app.py
        │  st.chat_input(...)
        ▼
[2] FRONTEND builds JSON: {query, history(last 5 turns)}        app.py
        │  requests.post("http://127.0.0.1:8000/query", ..., timeout=60)
        ▼
┌──────────────────────── BACKEND  src/api/server.py : query_agent() ───────────────────────┐
│                                                                                            │
│ [3] Guard: is the knowledge base ready?                                                    │
│        if not _vector_store:  ──► return HTTP 503 "upload a document first"                 │
│                                                                                            │
│ [4] SEMANTIC CACHE check (only if no history)        src/api/cache.py : query_cache.get()  │
│        embed query → cosine vs cached → ≥0.95 ? ──► RETURN cached answer (DONE, ~instant)   │
│                                                                                            │
│ [5] RETRIEVE relevant chunks               src/database/retriever.py : aretrieve_relevant() │
│        embed query → Qdrant nearest-4 with scores → keep only score ≥ 0.3                   │
│        │                                                                                    │
│        ├── if NOTHING ≥ 0.3 (and intent isn't CHAT) ──► refuse w/ chapter list (DONE)      │
│        ▼                                                                                    │
│ [6] BUILD sources list (page, topic, preview) from the kept chunks                         │
│                                                                                            │
│ [7] ROUTE intent              src/agents/router.py : route_query()  (Groq, temp 0.0)       │
│        returns "QUIZ" | "EXPLAIN" | "CHAT"                                                  │
│        │                                                                                    │
│        ├── "QUIZ"   ──► src/agents/quiz_agent.py    (Groq, temp 0.1, JSON)                  │
│        ├── "CHAT"   ──► canned greeting (no LLM)                                            │
│        └── "EXPLAIN"──► src/agents/concept_agent.py (Groq, temp 0.3, uses history+context)  │
│                                                                                            │
│ [8] WRAP result as QueryResponse{intent, response, sources}                                │
│                                                                                            │
│ [9] STORE in semantic cache (if stateless)          src/api/cache.py : query_cache.set()   │
│                                                                                            │
└──────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                            │ HTTP 200 + JSON
                                            ▼
[10] FRONTEND renders the answer                                app.py
        - EXPLAIN/CHAT → markdown + "Sources" expander (page numbers)
        - QUIZ → expandable question cards with radio options
        - append assistant reply to st.session_state.messages
```

### The same flow in plain English (your spoken script)
1. The student types a question; Streamlit grabs it.
2. The frontend packages the question **plus the last 5 turns of chat** and POSTs it to the backend's `/query`.
3. The backend first checks the knowledge base exists; if not, it returns a 503.
4. It checks the **semantic cache** — *have I answered something that means the same thing before?* If yes (≥0.95 similarity), it returns that instantly and we're done.
5. Otherwise it **retrieves**: embeds the question, asks Qdrant for the 4 nearest chunks with their scores, and keeps only those scoring ≥0.3. If nothing clears the bar, it **refuses** and lists the loaded chapters.
6. It builds a **sources** list (page number, topic, a short preview) from the surviving chunks.
7. The **Router** (a quick zero-temperature LLM call) decides: quiz, explain, or chat.
8. It dispatches to the right specialist — the **Quiz Agent** (strict JSON), the canned greeting, or the **Concept Agent** (explanation using context + history).
9. It wraps everything as a `QueryResponse`, **caches** it (if the query was stateless), and returns it.
10. The frontend renders it — a chat bubble with sources, or interactive quiz cards.

### How many network calls does one question make? (a sharp question you might get)
- 1 embedding (local, query) for the cache check
- 1 embedding (local, query) for retrieval *(could be deduplicated — a nice optimization to mention)*
- 1 Qdrant search (network)
- 1 Groq call for routing (network)
- 1 Groq call for the answer (network)

So a **cache miss** is ~2 LLM round-trips + 1 vector search; a **cache hit** is basically 1 local embedding and nothing else. That's why the cache is valuable.

---

## 3.2 Flow B — A student uploads a PDF

### Text flowchart
```
[1] STUDENT picks a PDF + clicks "Add to Knowledge Base"        app.py (sidebar)
        │  requests.post("/upload", files={...})
        ▼
[2] BACKEND /upload                              src/api/server.py : upload_document()
        ├── reject if not .pdf  ──► HTTP 400
        ├── save bytes to data/raw/<filename>
        ├── background_tasks.add_task(ingest_file, file_path)
        └── RETURN immediately: "Ingestion started in background"   (HTTP 200, fast)
                                            │
                 (server keeps serving other requests; ingest runs in the background)
                                            ▼
[3] BACKGROUND  ingest_file()                    src/api/server.py
        │
        ├── load_single_document(path)           src/ingestion/pdf_loader.py   (1 Document per page)
        ├── chunk_documents(docs)                src/ingestion/chunker.py      (~400-token chunks + metadata)
        ├── delete_by_source(path)               src/database/vector_store.py  (remove old copy → no dupes)
        ├── create_vector_db(chunks)             src/database/vector_store.py  (embed all + upsert to Qdrant)
        ├── _vector_store = get_vector_store()   (reconnect now that data exists)
        └── query_cache.clear()                  (old cached answers may be stale)
                                            │
                                            ▼
[4] QDRANT CLOUD now contains the new chunks; /knowledge-base reflects them
        │
        ▼
[5] FRONTEND sidebar (on next rerun) shows "Chapters loaded: N"  app.py  (polls /knowledge-base)
```

### Plain-English script
1. The student selects a PDF and clicks upload; Streamlit POSTs the file bytes to `/upload`.
2. The backend rejects non-PDFs, saves the file to `data/raw/`, schedules a **background task**, and **returns immediately** so the UI doesn't hang.
3. In the background: load the PDF (per page) → split into ~400-token chunks with page/topic metadata → **delete any old chunks from this same file** (idempotency) → **embed and upload** the new chunks to Qdrant → reconnect the vector store → clear the now-stale cache.
4. Qdrant now holds the new chapter.
5. On the next sidebar refresh, "Chapters loaded" ticks up. (Embedding takes ~20–40s on CPU, so it appears after a short delay — the upload call itself was instant.)

### Why the upload returns before ingestion finishes (a deliberate design point)
Embedding a whole chapter on CPU is slow (tens of seconds). If `/upload` waited for it, the HTTP request might time out and the UI would freeze. Instead we **acknowledge fast and process asynchronously** — a classic "accept now, process later" pattern. The trade-off: there's a window where the file is uploaded but not yet queryable. The sidebar's chapter count is how the student knows it finished.

---

## 3.3 Flow C — The "refusal" path (no relevant content)

This is worth knowing cold because it's your **anti-hallucination showcase.**

```
Student: "explain acid bases and salts"   (but only graph notes are loaded)
   │
   ▼
retrieve → Qdrant returns 4 chunks, top score ≈ 0.10
   │
   ▼
filter score ≥ 0.3  ──►  ALL dropped  ──►  retrieved_docs == []
   │
   ▼
intent = EXPLAIN (not CHAT) AND retrieved_docs is empty
   │
   ▼
build_no_context_message()  → reads loaded chapters from Qdrant
   │
   ▼
RETURN: "I couldn't find anything about that in your uploaded notes.
         Currently loaded chapters:
         - Graphs_Fable5.pdf
         Try asking about one of these, or upload the relevant PDF."
```
No LLM call is wasted, and the student gets an honest, helpful answer instead of a confident hallucination.

Continue to [4_EDGE_CASES_AND_SCALING.md](4_EDGE_CASES_AND_SCALING.md).
