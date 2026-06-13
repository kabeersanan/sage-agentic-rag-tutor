# 4. Edge Cases, Bottlenecks & Hardening

> Goal: be able to honestly say *"here's where this breaks, here's why, and here's exactly what I'd do about it."* Admitting limits **with a fix** is what separates senior answers from junior ones.

---

## 4.1 The biggest bottleneck: CPU embeddings

**Where:** `get_embedding_function()` → `all-mpnet-base-v2` running on CPU. Used in three places: ingestion, query retrieval, and the cache.

**Why it's slow:**
- The model takes **~12.6 seconds to load** (now a one-time cost thanks to the singleton).
- Embedding text is **CPU-bound** — it holds Python's GIL (Global Interpreter Lock), so while one embedding runs, other Python work on that process waits.
- Ingesting a chapter embeds dozens of chunks → **20–40 seconds** of CPU work.

**Symptoms under load:** during a big upload, query latency rises because the background embedding hogs the CPU/GIL.

**Fixes (in order of effort):**
1. **Move embeddings off the request path** (already partly done): the cache offloads embedding with `asyncio.to_thread`, and uploads run as background tasks.
2. **Use a GPU** for the embedding model → 10–50× faster.
3. **Use a hosted embedding API** (e.g., a managed embedding endpoint) so the app server does no heavy math.
4. **Switch to a smaller model** like `all-MiniLM-L6-v2` (384-dim, ~5× faster) if a small quality drop is acceptable. *(Caveat: changing the model changes the vector dimension, so you must wipe and rebuild the Qdrant collection at the new size.)*

---

## 4.2 Single process, in-memory state — the multi-worker trap

**Where:** two pieces of state live **inside one Python process**:
- `_vector_store` (the connection handle) in `server.py`
- `query_cache` (the semantic cache) in `cache.py`

**Why it breaks at scale:** the moment you run **more than one** uvicorn worker (e.g., `--workers 4`) or multiple server machines:
- Each worker has its **own separate cache** → cache hit rate drops, and a "clear cache" on one worker doesn't clear the others.
- Each worker re-loads the **12.6s embedding model** → slow, memory-heavy startup ×N.

**Fixes:**
1. **Shared cache:** move the semantic cache to **Redis** (or a managed vector cache) so all workers share it and it survives restarts.
2. **Shared embeddings:** run the embedding model as its **own service** (one model, many app workers call it), or use a hosted embedding API.
3. **Stateless app servers:** because the backend doesn't store chat history (the frontend does), the app servers are *already* horizontally scalable once the cache/embeddings are externalized. This is a strong point — say *"my backend is stateless per-request, so it scales horizontally; the only shared state is the cache, which I'd externalize to Redis."*

---

## 4.3 Background ingestion runs in-process

**Where:** `/upload` uses FastAPI `BackgroundTasks`, which run **inside the web server process**.

**Why it's a problem at scale:** a heavy CPU task (embedding) inside the web server competes with live request handling. If the server crashes mid-ingest, the task is lost with no retry.

**Fix:** move ingestion to a **dedicated task queue** — Celery or RQ with Redis, or a cloud queue. Benefits: isolation (embedding can't slow the API), retries on failure, and visibility into job status. You could then add a `GET /upload/status/{job_id}` endpoint so the UI shows real progress instead of polling chapter counts.

---

## 4.4 Rate limits and external dependencies

**Where:** every `/query` makes **2 Groq calls** (router + answer) and **1 Qdrant call**.

**Risks:**
- **Groq rate limits / outages:** under heavy traffic you can hit per-minute token limits, or Groq could be down.
- **Qdrant network latency:** each round-trip to Qdrant Cloud is ~1–2s from a distant region.

**Current handling:** the whole `/query` body is wrapped in `try/except` that returns a clean HTTP 500 with the error message instead of crashing.

**Hardening:**
1. **Retries with backoff** on transient Groq/Qdrant failures.
2. **Combine the two LLM calls:** the router + answer could sometimes be one call, or the router could be replaced by a cheap local classifier to cut latency and cost.
3. **Circuit breaker / fallback:** if Groq is down, return a graceful "try again shortly" instead of a 500.
4. **Deduplicate the query embedding:** today the cache check and retrieval each embed the query separately — compute it once and reuse it.

---

## 4.5 Security & robustness gaps (be honest, then fix)

| Gap | Risk | Fix |
|---|---|---|
| `CORS allow_origins=["*"]` | Any website can call your API | Restrict to your real frontend domain |
| **No authentication** on any endpoint | Anyone can query, upload, or **delete the whole knowledge base** | Add API keys / JWT auth; protect `DELETE /knowledge-base` |
| **No rate limiting** | One user can exhaust your Groq quota / run up cost | Add per-IP rate limiting (e.g., slowapi) |
| **No upload size/scan limits** | Huge or malicious PDFs | Cap file size; validate content; sandbox parsing |
| Secrets in `.env` | Key leakage if committed | Already gitignored; use a secrets manager in prod |
| `delete_knowledge_base` is one click | Accidental data loss | UI has a confirm checkbox; backend could add a soft-delete/snapshot |

---

## 4.6 Functional edge cases the code already handles (good to cite)

- **Empty / off-topic query** → relevance threshold returns `[]` → polite refusal listing loaded chapters. (No hallucination.)
- **Knowledge base not loaded** → `/query` returns **503** with a clear message; frontend shows it.
- **Non-PDF upload** → `/upload` returns **400** "Only PDF files are supported."
- **LLM returns broken JSON for a quiz** → `clean_json_text` + `try/except` → friendly fallback question instead of a crash.
- **Qdrant unreachable at startup** → `lifespan` `try/except` keeps the server alive; `/health` reports the problem.
- **Duplicate upload of the same file** → `delete_by_source` makes re-upload replace, not duplicate.
- **Stale answers after new upload** → cache is cleared on every KB change.
- **Slow stats call** → frontend uses fast `/health` for connectivity so it never falsely shows "offline."
- **Windows `localhost` IPv6 trap** → frontend uses explicit `127.0.0.1`.

---

## 4.7 The "scale to 100,000 users" answer (a complete spoken script)

> "Today it's a single-process app with an in-memory cache and CPU embeddings — perfect for a demo, not for 100k users. To scale, I'd do five things:
>
> **1. Externalize the heavy compute.** Move the embedding model to a GPU service or a hosted embedding API so the web servers do no heavy math.
>
> **2. Make the app servers stateless and horizontal.** They're already stateless per-request (chat history lives in the client), so I can run many copies behind a load balancer. The only shared state is the cache.
>
> **3. Externalize the cache to Redis** so all servers share one semantic cache that survives restarts.
>
> **4. Move ingestion to a real task queue** (Celery/RQ) with retries and a job-status endpoint, instead of in-process background tasks.
>
> **5. Add the production guardrails:** authentication (especially on delete), per-user rate limiting to protect my Groq quota, retries/backoff and a circuit breaker for the LLM and vector DB, and tightened CORS.
>
> Qdrant Cloud already scales to millions of vectors, and Groq handles the inference, so the bottleneck is really my app tier and the embedding compute — both of which the above addresses. I'd also add monitoring (latency, cache-hit rate, error rate) so I know where the next bottleneck is."

---

## 4.8 Quick "cost & latency budget" cheat-sheet

| Operation | Rough cost | Dominated by |
|---|---|---|
| Cache hit | ~30–60ms | one local query embedding |
| Cache miss (full answer) | ~1–3s | 2 Groq calls + 1 Qdrant search |
| Upload acknowledgement | instant | just saving the file |
| Full ingestion of a chapter | ~20–40s | CPU embedding of all chunks |
| Cold server start | ~12s | one-time embedding-model load |

Continue to [5_Q_AND_A.md](5_Q_AND_A.md).
