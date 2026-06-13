# 🎯 Sage AI Tutor — Interview Master Guide

This is your "defend-the-project" handbook. It is split into five focused files so each one stays readable. Read them in order the first time, then jump around to revise.

| # | File | What it covers |
|---|------|----------------|
| 1 | [1_ARCHITECTURE.md](1_ARCHITECTURE.md) | The big picture: every technology, what it does, and **why** it was chosen over alternatives. |
| 2 | [2_IMPLEMENTATION.md](2_IMPLEMENTATION.md) | Every feature, explained step-by-step, with the exact files and functions and code snippets.
 |
| 3 | [3_DATA_FLOW.md](3_DATA_FLOW.md) | Trace a question and a PDF upload end-to-end, with text flowcharts. |
| 4 | [4_EDGE_CASES_AND_SCALING.md](4_EDGE_CASES_AND_SCALING.md) | Where it breaks under load, how errors are handled, and how to scale to 100k users. |
| 5 | [5_Q_AND_A.md](5_Q_AND_A.md) | A large bank of likely interview questions with simple, STAR-style spoken answers. |

---

## ⚠️ READ THIS FIRST — The README is out of date (important interview trap)

The repo's `README.md` describes an **older version** of this project. The **code you actually have now is different**. If you claim what the README says, an interviewer who reads your code will catch the mismatch. Here is the truth table — **memorize this**:

| README says (OLD) | Your code actually does (NEW) |
|---|---|
| Vector DB is **ChromaDB** (local) | Vector DB is **Qdrant Cloud** (hosted) |
| "L2-Normalized Confidence Score" with formula `1/(1+distance)` | **No** confidence scoring. You use a **raw cosine similarity threshold of 0.3** to decide relevance. |
| Intents are `FACT` / `CONCEPT` | Intents are **`QUIZ` / `EXPLAIN` / `CHAT`** |
| Mentions an `evaluate.py` evaluation script | That file is **not in the current code** |
| No mention of caching | You have a **semantic (embedding-based) cache** |
| No mention of a web API | You have a full **FastAPI backend** that the Streamlit app talks to over HTTP |

**One-line way to explain this in an interview if asked:** *"The README documents an earlier iteration. The project evolved: I migrated from local ChromaDB to managed Qdrant Cloud, replaced the confidence-score idea with a cleaner relevance threshold, and split the app into a FastAPI backend plus a Streamlit frontend with a semantic cache. I should update the README."*

That answer turns a weakness into a strength (it shows the project evolved and you know it cold).

---

## The 30-second elevator pitch (say this when asked "tell me about your project")

> "Sage is an AI tutor for Class 10 students. You upload a textbook PDF, and you can chat with it — ask it to explain a concept or to quiz you. Under the hood it's a **RAG system** (Retrieval-Augmented Generation): instead of letting the AI make things up, I store the textbook in a vector database, find the most relevant passages for each question, and feed only those to the language model. I added an **agentic router** that first decides *what the student wants* — an explanation, a quiz, or just a greeting — and routes to a specialist for each. It's built as a **FastAPI backend** with a **Streamlit frontend**, uses **Groq's Llama-3.3-70B** for fast generation, **local sentence-transformer embeddings**, and **Qdrant Cloud** for vector storage. I also added a semantic cache and a relevance threshold so it refuses to answer when the topic isn't in the loaded notes."

---

## Your project in one diagram

```
                          ┌─────────────────────────┐
   Student types  ───────▶│   STREAMLIT FRONTEND     │   app.py
   a question             │   (chat UI + sidebar)    │
                          └───────────┬─────────────┘
                                      │ HTTP (JSON)
                                      ▼
                          ┌─────────────────────────┐
                          │    FASTAPI BACKEND       │   src/api/server.py
                          │  /query /upload /health  │
                          │  /knowledge-base         │
                          └───────────┬─────────────┘
                                      │
            ┌─────────────────────────┼──────────────────────────┐
            ▼                         ▼                          ▼
   ┌─────────────────┐     ┌────────────────────┐     ┌────────────────────┐
   │ SEMANTIC CACHE  │     │   RETRIEVAL LAYER  │     │   AGENT LAYER      │
   │ src/api/cache.py│     │ src/database/*.py  │     │ src/agents/*.py    │
   │ (skip work if   │     │ embed → Qdrant     │     │ Router → Concept   │
   │  seen before)   │     │ search → threshold │     │ or Quiz (Groq LLM) │
   └─────────────────┘     └─────────┬──────────┘     └────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────┐
                          │     QDRANT CLOUD         │   (hosted vector DB)
                          │  collection of chunks    │
                          └─────────────────────────┘
```

Now go to [1_ARCHITECTURE.md](1_ARCHITECTURE.md).
