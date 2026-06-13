# 5. Anticipated Interview Questions & Simple Answers

> How to use this file: read the question, try to answer **out loud first**, then check. The answers use **STAR** where useful (Situation, Task, Action, Result). Keep your spoken answers to ~30–60 seconds; the extra detail here is for your understanding.

Sections:
- A. High-level system design
- B. Deep code-level logic
- C. Trade-offs & "gotchas"
- D. Rapid-fire one-liners (memorize these)
- E. Likely curveballs

---

## A. High-Level System Design

### A1. "Walk me through your project."
*(Use the elevator pitch from the master guide, then offer the one-diagram view.)*
> "Sage is a RAG-based AI tutor. A student uploads a textbook PDF and chats with it — asking for explanations or quizzes. I split it into a **FastAPI backend** and a **Streamlit frontend**. When a question comes in, the backend embeds it, finds the most relevant textbook chunks in **Qdrant Cloud**, and an **agentic router** decides whether to explain, quiz, or just greet — then the right specialist uses **Groq's Llama-3.3-70B** to answer, grounded only in the retrieved text, with page citations. I added a **semantic cache** and a **relevance threshold** so it's fast and won't hallucinate when a topic isn't in the notes."

### A2. "What is RAG and why use it instead of just asking the LLM directly?"
> "RAG = Retrieval-Augmented Generation. Instead of trusting the model's memory, I store the textbook in a vector database, retrieve the few most relevant passages for each question, and give **only those** to the model as context. **Why:** it keeps answers grounded in the actual textbook (less hallucination), lets me **cite page numbers**, and means I can update knowledge by uploading a new PDF instead of retraining a model."

### A3. "What makes it *agentic* rather than plain RAG?"
> "Plain RAG has one path: retrieve, stuff into a prompt, answer. Mine has a **Router agent** that first classifies the user's intent — QUIZ, EXPLAIN, or CHAT — and dispatches to a **different specialist** for each, each with its own prompt and temperature. It's like a triage nurse sending you to the right doctor instead of one generalist handling everything."

### A4. "Why a separate backend and frontend instead of one Streamlit app?"
> **S/T:** "I needed a clean architecture that could grow. **A:** I put all the AI logic behind a FastAPI API and kept Streamlit as a thin client that just calls it. **R:** Now the UI can be replaced (React, mobile, a bot) without touching the AI logic, and I can scale the backend independently behind a load balancer. The frontend is the waiter, the backend is the kitchen."

### A5. "Why Qdrant Cloud over ChromaDB / FAISS / Pinecone?"
> "I actually **started with local ChromaDB** but it stores vectors in a folder on one machine, which doesn't work for a deployable, multi-machine setup. I migrated to **Qdrant Cloud** — it's managed and always-on, supports metadata filtering and payload indexes (which I use for idempotent deletes), and scales to millions of vectors. FAISS is just a library with no persistence/server; Pinecone is similar to Qdrant but I preferred Qdrant's open-source option and free tier."

### A6. "Why Groq + Llama-3 over OpenAI or Gemini?"
> "Latency is the deciding factor for a tutor — waiting 5 seconds breaks a student's concentration. Groq runs Llama-3 on special hardware with near-instant token generation. It's also cheaper than GPT-4 and uses an open model, so I'm not vendor-locked. I'd previously seen >2s latency and rate limits on a hosted Gemini setup, which pushed me to Groq for the real-time feel."

### A7. "How would you scale this to 100,000 users?"
*(Use the full script in [4_EDGE_CASES_AND_SCALING.md](4_EDGE_CASES_AND_SCALING.md) §4.7 — externalize embeddings, stateless horizontal app servers, Redis cache, task queue for ingestion, plus auth/rate-limit/retries.)*

---

## B. Deep Code-Level Logic

### B1. "How did you handle conversation state / memory?"
> **S:** "Follow-ups like 'when was *he* born?' need memory. **T:** Resolve pronouns without making the backend stateful. **A:** I keep the full chat in the **frontend's** `st.session_state` and send the **last 5 turns** with each request. The Concept Agent injects that history into its prompt. The backend stays **stateless** — it remembers nothing between calls. **R:** Natural follow-ups work, and because the server is stateless, I can run many copies without sticky sessions."

**Follow-up — "Why keep memory on the frontend?"**
> "Statelessness makes the backend horizontally scalable — any server can handle any request because all the context arrives in the request. The trade-off is a slightly larger request payload, which is cheap."

### B2. "How does retrieval decide what's relevant? Walk me through the threshold."
> "`aretrieve_relevant` embeds the question, asks Qdrant for the 4 nearest chunks **with their raw cosine scores**, and keeps only those scoring **≥ 0.3**. If all four are below 0.3, it returns an empty list and the system refuses to answer. I calibrated 0.3 from real data: genuine matches scored 0.6–0.74, off-topic questions scored under 0.2."

**Follow-up — "Why 0.3 specifically?"**
> "It's empirical, not a guess. I measured scores for in-corpus and out-of-corpus queries; real matches were ≥0.32, junk was ≤0.2, so 0.3 sits in the clean gap. It's a single config constant (`SIMILARITY_THRESHOLD`) so I can tune it."

### B3. "Your framework had a built-in score threshold. Why did you write your own filter?"
*(This is a gold-star answer — it shows depth.)*
> "LangChain's `score_threshold` **re-normalizes** the score with an opaque formula — it mapped a raw cosine of 0.106 up to about 0.55, so my threshold let junk through. I verified the actual numbers, realized I couldn't trust the framework's hidden mapping for a hard cutoff, and instead filtered on the **raw Qdrant cosine score** using `asimilarity_search_with_score`. Lesson: when a framework abstraction hides a number you depend on, measure it yourself."

### B4. "How does the semantic cache work, and how is it different from a normal cache?"
> "A normal cache matches exact text. Mine matches **meaning**: it embeds the query and compares it to past queries by cosine similarity using a vectorized NumPy dot product. If the best match is ≥ 0.95, it's effectively the same question and I return the stored answer — skipping both the vector search and the LLM. Eviction is FIFO at 100 entries. I only cache **stateless** queries (no chat history), and I **clear the cache whenever the knowledge base changes**, since old answers may be stale."

**Follow-up — "Why L2-normalize the vectors in the cache?"**
> "If both vectors are unit-length, their dot product **equals** cosine similarity — so I can use one fast `np.dot` over all cached vectors at once instead of computing norms each time."

### B5. "How do you stop the quiz generation from breaking when the LLM returns messy JSON?"
> "Three defenses. One: temperature 0.1 for strict formatting. Two: a regex cleaner, `clean_json_text`, that strips ```` ```json ``` ```` fences the model loves to add. Three: the parse is wrapped in `try/except` — if JSON still fails, I return a friendly fallback question instead of crashing. That cleaner is my most unit-tested function because flaky LLM JSON is the most fragile part."

### B6. "How does the PDF chunking work, and why not just cut every 1000 characters?"
> "I use a recursive splitter with a **priority list of separators**: paragraphs first, then lines, then sentences, down to characters as a last resort. So it breaks on natural boundaries instead of slicing a sentence in half. I also add **50-token overlap** between chunks so meaning isn't lost at the cut, and I attach **page and topic metadata** to each chunk for citations. Target size is ~400 tokens — big enough for context, small enough for precise retrieval."

### B7. "How did you make uploading a PDF twice not create duplicates?"
> **S:** "Re-uploading a file was appending a second copy of its chunks. **T:** Make ingestion idempotent. **A:** Before inserting, I call `delete_by_source(file_path)` to remove any existing chunks from that exact file, then insert the fresh ones. **R:** Re-upload now **replaces** instead of duplicating. The gotcha: Qdrant won't filter/delete by a field unless it's **indexed**, so I had to create a `keyword` payload index on `metadata.source` — that was a 400-error bug I debugged and fixed."

### B8. "Why is the upload endpoint async/background? What happens if I query right after uploading?"
> "Embedding a chapter takes 20–40s on CPU, so `/upload` saves the file, schedules a **background task**, and returns immediately — otherwise the request would time out. If you query during that window, retrieval just won't find the new chunks yet; the sidebar's 'Chapters loaded' count tells you when ingestion finished. Right after a fresh delete+upload, a query can briefly get a 503 until the vector store reconnects — which then resolves on its own."

### B9. "Where do you use async, and why does it matter here?"
> "The endpoints and agent calls are `async`. Our work is mostly **waiting** — on Groq and on Qdrant. Async lets one server handle other requests during that waiting instead of blocking. I also offload **CPU** work (embedding in the cache) with `asyncio.to_thread` so it doesn't freeze the event loop. Async helps with I/O waiting; threads help with CPU work — I use the right one for each."

### B10. "How are request and response shapes validated?"
> "Pydantic models in `schemas.py` define the contract — `QueryRequest`, `QueryResponse`, `Source`, etc. FastAPI validates incoming JSON against them automatically and rejects bad requests with a clear error, and it serializes responses the same way. It's my safety net against typo'd or malformed data, and it auto-generates API docs."

---

## C. Trade-offs & "Gotchas" In My Implementation

### C1. "What's the biggest performance bottleneck?"
> "CPU embeddings. The `all-mpnet-base-v2` model takes ~12.6s to load and embedding is CPU-bound. I fixed the repeated-load problem by making the model a **singleton** — load once, reuse — which also fixed a bug where a slow stats call made the UI falsely show 'offline.' To go further I'd put embeddings on a GPU or a hosted embedding API, or switch to a smaller model."

### C2. "Tell me about a tricky bug you fixed."
*(Pick whichever lands best; all are real.)*
> **The collection-name mismatch:** "The writer created a Qdrant collection with underscores (`sage_agentic_rag_tutor`) but the reader queried hyphens (`sage-agentic-rag-tutor`). That 404'd inside FastAPI's startup hook, which **crashed uvicorn before it bound port 8000** — so the frontend just said 'backend unreachable.' I traced it, **centralized the collection name in one config constant** so the two can't drift, and wrapped startup in try/except so a connection error reports via `/health` instead of killing the server."

> **The Windows `localhost` trap:** "The frontend said 'backend unreachable' even though it was running. On Windows, `localhost` can resolve to IPv6 `::1` while uvicorn binds IPv4 `127.0.0.1`. I changed the URLs to the explicit `127.0.0.1` and it connected."

> **The false 'offline' status:** "The sidebar checked connectivity using the **slow** `/knowledge-base` call (~5s), which exceeded the 5s timeout and falsely showed 'offline.' I switched connectivity to the fast `/health` endpoint (~0.2s) and loaded the chapter list separately with a longer timeout."

### C3. "Where could this hallucinate, and how did you prevent it?"
> "The danger is feeding the LLM irrelevant chunks. I saw it live: with only graph notes loaded, asking about acids/bases returned graph chunks at ~0.1 similarity and the model confidently explained graphs. I added **two layers of defense**: a retrieval **relevance threshold** (drop chunks < 0.3, refuse if none qualify), and a **grounding instruction** in the Concept prompt ('use ONLY the context; if it's not there, say so'). Now off-topic questions get an honest refusal that even lists the loaded chapters."

### C4. "What are the weaknesses of your semantic cache?"
> "It's **in-memory and per-process**, so multiple workers each have their own cache and a restart wipes it. The 0.95 threshold is conservative to avoid serving a subtly-different question's answer. And FIFO eviction isn't usage-aware — a popular entry can be evicted. At scale I'd move it to Redis and consider LRU eviction."

### C5. "What would you refactor first if you came back to this?"
> "A few things: (1) fix the **stale README** to match the current Qdrant/cache implementation; (2) **deduplicate the query embedding** — the cache check and retrieval embed the query twice; (3) move ingestion to a **task queue** with retries and a status endpoint; (4) add **auth and rate limiting**, especially protecting the delete endpoint; (5) add **automated evaluation** so I can measure retrieval quality and tune the threshold objectively."

### C6. "Your backend makes two LLM calls per query (router + answer). Isn't that wasteful?"
> "It's a deliberate trade-off: a tiny zero-temperature routing call buys me clean separation and specialist prompts, which improves answer quality. But yes, it adds latency and cost. I could replace the router with a cheap local classifier or merge it into a single structured call. For a demo the clarity was worth it; at scale I'd optimize it."

---

## D. Rapid-Fire One-Liners (memorize)

- **Embedding model & size:** `all-mpnet-base-v2`, **768 dimensions**, runs locally on CPU.
- **Distance metric:** **Cosine** (angle between vectors = same-topic-ness).
- **Relevance threshold:** **0.3** raw cosine (`SIMILARITY_THRESHOLD`).
- **Cache hit threshold:** **0.95** cosine; FIFO; max **100** entries.
- **Chunk size / overlap:** ~**400 tokens** / **50 tokens**.
- **LLM:** Groq **Llama-3.3-70B-versatile**.
- **Temperatures:** Router **0.0**, Quiz **0.1**, Concept **0.3**.
- **History window:** last **5 turns** sent from frontend.
- **Endpoints:** `POST /query`, `POST /upload`, `GET/DELETE /knowledge-base`, `GET /health`.
- **Idempotency:** `delete_by_source` + a **keyword payload index** on `metadata.source`.
- **Backend state:** **stateless per request** (chat history lives in the frontend).
- **Two anti-hallucination layers:** retrieval threshold + grounding prompt.

---

## E. Likely Curveballs

### E1. "What if two students upload PDFs at the same time?"
> "Both uploads are accepted instantly and each schedules its own background ingestion. They share one Qdrant collection, and `create_vector_db` appends, so both land correctly. The risk is CPU contention slowing things down, since embedding is in-process — the fix is an external task queue so ingestions are isolated and serialized/parallelized deliberately."

### E2. "How do you know your retrieval is actually good?"
> "Right now I validated it by **measuring similarity scores** on known in-corpus vs out-of-corpus queries and confirming the threshold cleanly separates them. The honest gap is I don't have an automated eval set — I'd add 'golden' question/expected-source pairs and track retrieval precision so I can tune the threshold and chunking objectively."

### E3. "What happens if Groq is down?"
> "Currently the `/query` try/except returns a clean HTTP 500 and the frontend shows an error rather than crashing. To harden it I'd add retries with backoff and a circuit breaker that returns a graceful 'try again shortly,' and possibly a fallback model."

### E4. "Why not fine-tune a model instead of RAG?"
> "Fine-tuning bakes knowledge into weights — expensive, slow to update, and it still hallucinates and can't cite sources. RAG lets me **update knowledge by uploading a PDF**, keeps answers grounded in the exact text, and gives **page citations**. For a tutor that must be trustworthy and updatable, RAG is the right tool."

### E5. "If I remove the cache, what changes?"
> "Correctness is unchanged — the cache is purely an optimization. Repeated/paraphrased questions would just be slower and cost more, since every one would hit the vector search and two LLM calls again."

### E6. "Explain your project to a non-technical person."
> "It's a study buddy that reads your textbook for you. You give it the PDF, then ask it questions in plain language. It finds the exact pages that answer you and explains them simply — or quizzes you. And if you ask about something that's not in your book, it tells you honestly instead of guessing."

---

## Final prep checklist
- [ ] I can give the 30-second pitch without notes.
- [ ] I can draw the architecture diagram from memory.
- [ ] I can trace a `/query` end-to-end naming each file.
- [ ] I can trace an upload end-to-end and explain why it's a background task.
- [ ] I know all the numbers in section D cold.
- [ ] I can tell 2–3 real bug stories (collection-name crash, threshold/hallucination, idempotency index, Windows localhost).
- [ ] I can give the "scale to 100k" answer.
- [ ] I will proactively mention the README is outdated and say what actually changed.

Good luck — you built a genuinely solid system. Own it.
