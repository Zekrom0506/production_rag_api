# Production RAG API

A production-style FastAPI backend for a Retrieval-Augmented Generation (RAG) chatbot, built with FastAPI, LangGraph, LangChain, Groq, and a Postgres vector store (pgvector).

Rather than a simple prompt-and-response wrapper, the service runs an agentic retrieval loop with self-correction, plus the security, caching, and observability layers you'd expect from a real backend rather than a notebook demo.

## Features

**Retrieval-Augmented Generation**
- Agentic RAG pipeline orchestrated with LangGraph (a state-machine framework for LLM apps).
- Vector search over documents stored in Postgres via `pgvector`.
- Documents are embedded locally with a HuggingFace embedding model (`BAAI/bge-small-en-v1.5`) — no external embedding API required.
- Self-correcting retrieval: retrieved documents are graded for relevance by an LLM; if they fall short, the query is automatically rewritten and retrieval retried, up to a configurable limit.
- Graceful degradation: if no relevant documents are found, the API returns an honest "I couldn't find that" response instead of hallucinating.

**LLM layer**
- Groq as the inference provider, using Llama 3.1 / 3.3 models.
- Primary + fallback model setup: if the primary model call fails, the app retries automatically with a fallback model.
- A separate, cheaper model handles relevance grading to keep cost and latency down.

**Security**
- Prompt-injection detection on incoming messages (regex-based screening for common jailbreak patterns) before anything reaches the LLM.
- PII detection and masking (emails, phone numbers, credit card numbers, Aadhaar numbers) on both input and output.
- Output validation to catch leaked secrets or harmful content before a response is returned.

**Performance & reliability**
- In-memory response caching with TTL, keyed on a normalized/hashed query.
- Per-client rate limiting via `slowapi`.
- Configurable retry logic across LLM calls and retrieval.

**Observability**
- Structured JSON logging.
- Custom metrics (total requests, error rate, average latency, cache hit rate, token usage) exposed via `/metrics`.
- LangSmith tracing throughout the pipeline for request-level debugging.
- `/health` endpoint for uptime checks and container orchestration.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Main RAG chat endpoint |
| GET | `/health` | Health check |
| GET | `/metrics` | Runtime metrics |
| GET | `/cache/stats` | Cache performance stats |

## Tech stack

FastAPI · LangGraph · LangChain · Groq · Postgres + pgvector · HuggingFace sentence-transformers · Docker

## Getting started

### Prerequisites
- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Postgres database with the `pgvector` extension (e.g. [Supabase](https://supabase.com/))
- A [Groq](https://console.groq.com/) API key

### Setup

```bash
git clone https://github.com/Zekrom0506/production_rag_api.git
cd production_rag_api

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# then fill in GROQ_API_KEY, DATABASE_URL, and (optionally) LANGCHAIN_API_KEY
```

### Run locally

```bash
uv run uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Run with Docker

```bash
docker compose up --build
```

### Run tests

```bash
uv run pytest
```

## Configuration

All configuration is environment-driven (see `.env.example`). Key variables:

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key (required) | — |
| `DATABASE_URL` | Postgres connection string (required) | — |
| `PRIMARY_MODEL` | Primary LLM | `llama-3.1-8b-instant` |
| `FALLBACK_MODEL` | Fallback LLM | `llama-3.3-70b-versatile` |
| `RATE_LIMIT` | Requests per client | `20/minute` |
| `CACHE_TTL_SECONDS` | Response cache TTL | `300` |
| `MAX_RETRIES` | Retry limit for LLM/retrieval calls | `3` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `true` |

## Deployment

A `render.yaml` is included for one-click infrastructure-as-code deployment on [Render](https://render.com/). Secrets (`GROQ_API_KEY`, `DATABASE_URL`, `LANGCHAIN_API_KEY`) are configured directly in the Render dashboard and never committed to source control.

## Project structure

```
app/
├── main.py         # FastAPI app, routes, lifespan
├── agent.py        # LangGraph RAG agent
├── security.py     # Prompt-injection + PII protection
├── cache.py        # Response caching
├── monitoring.py   # Logging + metrics
├── config.py        # Settings management
├── ingestion.py     # Document ingestion into the vector store
└── vectorstore.py   # Postgres/pgvector setup
tests/                # Unit tests
```

## License

No license has been chosen yet — all rights reserved by default. Add a `LICENSE` file if you'd like to make reuse terms explicit.
