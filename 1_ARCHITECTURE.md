# 1. System Architecture & Component Breakdown

> Goal of this file: by the end you can explain, in plain English, **what each piece does** and **why it's there** — without looking at notes.

---

## 1.1 The simplest possible overview

Imagine a **very well-read librarian** who has memorized one textbook.

- You ask a question.
- The librarian doesn't answer from their own imagination. They first **walk to the shelf and pull out the 4 most relevant pages** about your question.
- They read those pages, then **explain the answer in their own simple words**, and tell you which page it came from.
- If your question isn't covered anywhere in the book, they honestly say *"that's not in this textbook"* instead of making something up.

That is literally what this system does. The "shelf" is the **vector database (Qdrant)**, "pulling the relevant pages" is **retrieval**, and "explaining in simple words" is the **LLM (Llama-3 on Groq)**. The whole pattern is called **RAG = Retrieval-Augmented Generation**.

We add one twist: before answering, a **Router** decides whether you want an *explanation*, a *quiz*, or are just saying *hello* — and sends you to the right specialist. That twist is what makes it **"Agentic" RAG** instead of plain RAG.

---

## 1.2 The two halves: Frontend and Backend

The project is split into two programs that run separately and talk over HTTP:

1. **Frontend (`app.py`)** — a **Streamlit** web app. This is the chat window the student sees. It has zero AI logic. Its only job is: draw the UI, take what the user types, send it to the backend, and display whatever comes back.

2. **Backend (`src/api/server.py`)** — a **FastAPI** web server. This is the brain. It holds all the logic: retrieval, routing, the LLM calls, caching, and managing the knowledge base.

**Why split them?** (great interview answer)
- **Separation of concerns:** The UI can change (Streamlit today, a React app or a mobile app tomorrow) without touching the AI logic. The backend exposes a clean API; any client can use it.
- **Independent scaling:** If 1000 students hit you at once, you can run many copies of the backend behind a load balancer while keeping one simple frontend.
- **Reusability:** The same `/query` endpoint could power a website, a WhatsApp bot, or a mobile app.

> Analogy: The frontend is the **waiter** (takes your order, brings your food). The backend is the **kitchen** (does the actual cooking). You can redecorate the dining room without rebuilding the kitchen.

---

## 1.3 Component-by-component: what each technology does and WHY

### Streamlit (Frontend) — `app.py`
- **What it does:** Renders the chat interface, a sidebar showing system status and loaded chapters, a PDF uploader, and "clear chat" / "delete knowledge base" buttons. It calls the backend with the `requests` library.
- **Why chosen over alternatives (e.g., React):**
  - **Speed of building:** Streamlit lets you build a usable web UI in pure Python with ~150 lines. A React frontend would need a separate JavaScript project, build tooling, and API plumbing.
  - **Right tool for the job:** This is a demo/portfolio project, not a consumer product with 50 screens. Streamlit is perfect for "data app" UIs.
  - **Trade-off you should admit:** Streamlit re-runs the whole script top-to-bottom on every interaction, and state lives in `st.session_state`. It's not built for high-concurrency production UIs. For a real product you'd move to React/Next.js. (Knowing this trade-off scores points.)

### FastAPI + Uvicorn (Backend) — `src/api/server.py`
- **What it does:** Defines the HTTP endpoints (`/query`, `/upload`, `/health`, `/knowledge-base`). **Uvicorn** is the actual server program that runs FastAPI and listens on port 8000.
- **Why chosen over alternatives (e.g., Flask, Django):**
  - **Async support:** FastAPI is built on `async`/`await`. Our work is full of "waiting" (waiting for the LLM, waiting for the database). Async lets one server handle many requests at once instead of freezing on each one. (More on this in file 4.)
  - **Automatic validation + docs:** FastAPI uses **Pydantic** to validate request/response shapes automatically, and auto-generates interactive API docs at `/docs`.
  - **Lightweight:** Django is a heavy "batteries-included" framework (built-in ORM, admin, templates) we don't need. FastAPI is lean and API-first.

### Pydantic (Data contracts) — `src/api/schemas.py`
- **What it does:** Defines the exact shape of data going in and out as Python classes (`QueryRequest`, `QueryResponse`, `Source`, etc.). If a request is missing a field or has the wrong type, FastAPI rejects it automatically with a clear error.
- **Why:** It's your **contract** between frontend and backend. No silent bugs from a typo'd field name. It also documents the API for free.

### Groq + Llama-3.3-70B (The LLM) — used in `src/agents/*.py`
- **What it does:** Groq is a cloud service that runs the Llama-3.3-70B language model **extremely fast** on special hardware (LPUs). This is the component that actually writes the explanations and quiz questions.
- **Why chosen over alternatives (e.g., OpenAI GPT-4, Google Gemini):**
  - **Latency (the #1 reason):** Groq is famous for near-instant token generation. For a *tutor*, waiting 5 seconds for an answer breaks the student's concentration. Fast feels "real-time."
  - **Cost:** Llama-3 on Groq is cheap/free-tier friendly compared to GPT-4.
  - **Open model:** Llama-3 is open-weight, so you're not locked into one vendor.
  - *(Note: the README's story about migrating from Gemini to Groq for latency is a legitimate talking point — that decision is real even if other README details are stale.)*

### LangChain (Orchestration glue) — used in `src/agents/*.py` and `src/database/*.py`
- **What it does:** LangChain is the "plumbing" that connects prompts → model → output parsing in a clean pipeline. You see it as `chain = prompt | llm | parser`. It also provides the `QdrantVectorStore` wrapper and the HuggingFace embeddings wrapper.
- **Why:** It saves you from writing boilerplate (formatting prompts, parsing responses, talking to the vector DB). The `|` (pipe) syntax makes each agent a readable 3-line pipeline.
- **Trade-off to admit:** LangChain adds abstraction layers that can hide what's really happening and occasionally make debugging harder. For this project the convenience wins.

### Sentence-Transformers `all-mpnet-base-v2` (Embeddings) — `src/database/vector_store.py`
- **What it does:** An "embedding model" turns a piece of text into a list of **768 numbers** (a "vector") that captures its *meaning*. Two texts about the same topic produce vectors that are close together. This runs **locally on your CPU**.
- **Why chosen:**
  - **Zero cost / offline:** It runs on your machine, so embedding thousands of chunks costs nothing and needs no API.
  - **Quality:** `all-mpnet-base-v2` is one of the best general-purpose open embedding models for its size.
  - **Trade-off to admit (important):** It's **slow on CPU** — the model takes ~12 seconds to load and embedding many chunks is the main latency bottleneck. A paid embedding API or a GPU would be faster. This is a known, deliberate cost/speed trade-off. (Covered in file 4.)

### Qdrant Cloud (Vector Database) — `src/database/vector_store.py` & `retriever.py`
- **What it does:** Stores all the textbook chunks as vectors and lets you ask *"give me the 4 chunks whose meaning is closest to this question."* That "closest by meaning" search is called **similarity search / nearest-neighbor search**.
- **Why chosen over alternatives (e.g., ChromaDB, FAISS, Pinecone):**
  - **Managed & always-on:** Qdrant Cloud is hosted, so your data survives server restarts and is reachable from anywhere. Local ChromaDB lives in a folder on one machine.
  - **Production-shaped:** It supports payload indexes, metadata filtering, and scales to millions of vectors — closer to what a real product needs.
  - **Why you migrated FROM Chroma (your real story):** ChromaDB stored vectors in a local folder, which doesn't work when you want a deployable, multi-machine setup. Qdrant Cloud decouples storage from the app server.

### NumPy (Math for the cache) — `src/api/cache.py`
- **What it does:** Does the fast vector math (dot products) for the semantic cache.
- **Why:** It's the standard, fast way to do array math in Python. A pure-Python loop would be far slower.

### The `src/` package layout (why the code is organized this way)
```
src/
├── api/          # The web layer (FastAPI server, request/response schemas, cache)
│   ├── server.py
│   ├── schemas.py
│   └── cache.py
├── agents/       # The "thinking" layer (LLM-powered router + specialists + prompts)
│   ├── router.py
│   ├── concept_agent.py
│   ├── quiz_agent.py
│   └── prompts.py
├── database/     # The "memory" layer (embeddings, vector store, retrieval)
│   ├── vector_store.py
│   └── retriever.py
├── ingestion/    # The "reading" layer (load PDFs, split into chunks)
│   ├── pdf_loader.py
│   └── chunker.py
└── config.py     # All the settings/constants in one place
```
- **Why this matters in an interview:** It shows you understand **layered architecture**. Each folder has one job. The `api` layer never does math; the `database` layer never talks HTTP. This makes the code testable and easy to change. Say: *"I organized it by responsibility — ingestion, storage/retrieval, reasoning agents, and the API — so each layer can be tested and swapped independently."*

---

## 1.4 The "stack at a glance" table (memorize this)

| Layer | Technology | One-line reason it was chosen |
|---|---|---|
| Frontend | **Streamlit** | Build a web UI in pure Python, fast. |
| Backend API | **FastAPI + Uvicorn** | Async (handles waiting efficiently) + auto-validation + auto-docs. |
| Data contracts | **Pydantic** | Guarantees request/response shapes; no silent bugs. |
| LLM | **Groq — Llama-3.3-70B** | Sub-second generation = real-time tutor feel; cheap; open model. |
| Orchestration | **LangChain** | Clean `prompt \| llm \| parser` pipelines; vector-store wrappers. |
| Embeddings | **sentence-transformers all-mpnet-base-v2** | Free, offline, high quality (trade-off: slow on CPU). |
| Vector DB | **Qdrant Cloud** | Managed, always-on, production-shaped, scales. |
| Cache math | **NumPy** | Fast cosine similarity for the semantic cache. |

---

## 1.5 What makes this "Agentic" (a question you WILL get)

Plain RAG = retrieve → stuff into one prompt → answer. **One path for everything.**

Your system is **agentic** because it has a **Router agent** that makes a *decision* first, then dispatches to one of **three different specialist behaviors**, each with its own prompt and its own settings:

- **Router** (`router.py`) — classifies intent into `QUIZ`, `EXPLAIN`, or `CHAT`. Runs at **temperature 0.0** (zero randomness — we want a consistent, predictable label, not creativity).
- **Concept Agent** (`concept_agent.py`) — explains concepts using analogies and the chat history. Runs at **temperature 0.3** (a little creativity for friendly explanations).
- **Quiz Agent** (`quiz_agent.py`) — produces a strict JSON quiz. Runs at **temperature 0.1** (almost no randomness — we need valid, parseable JSON).

> Analogy: It's like a **hospital triage nurse**. You don't see a random doctor. The nurse (Router) first decides if you need a cardiologist (Concept) or a lab test (Quiz), and sends you to the right one. Each specialist has different tools and a different style.

Continue to [2_IMPLEMENTATION.md](2_IMPLEMENTATION.md).
