# What I Built — Production RAG API (Interview Talking Points)

A production-style FastAPI backend that wraps a Retrieval-Augmented Generation (RAG) chatbot, built with FastAPI, LangGraph, LangChain, Groq, and a Postgres vector store.

## Core RAG pipeline

- Built an agentic RAG pipeline using LangGraph (a state-machine framework for LLM apps), not just a single prompt-and-response call.
- Vector search over documents stored in PGVector (Postgres + pgvector extension) hosted on Supabase.
- Documents are embedded locally using a HuggingFace embedding model (BAAI/bge-small-en-v1.5) — no external embedding API needed.
- Self-correcting retrieval loop: after retrieving documents, an LLM "grades" them for relevance. If they're not relevant enough, the system automatically rewrites the user's query and retries retrieval, up to a configurable number of retries.
- Graceful degradation: if no relevant documents are ever found, the system returns an honest "I couldn't find that" answer instead of hallucinating.

## LLM layer

- Uses Groq as the LLM provider (fast inference) with Llama 3.1/3.3 models.
- Primary + fallback model setup: if the primary model call fails, the app automatically retries with a fallback model before giving up.
- Separate, cheaper model used just for grading document relevance (keeps cost/latency down vs. using the main model for everything).

## Security

- Prompt injection detection: input is scanned against regex patterns for common jailbreak attempts ("ignore previous instructions", "pretend you are", "reveal your system prompt", etc.) and blocked before reaching the LLM.
- PII detection and masking: emails, phone numbers, credit card numbers, and Aadhaar numbers are automatically detected and redacted — both on the way into the LLM and on the way out, so PII never gets logged or echoed back.
- Output validation: LLM responses are scanned for leaked secrets (API keys, passwords) or harmful content patterns before being returned to the client.

## Performance / reliability

- In-memory response caching with TTL (time-to-live), keyed by a normalized/hashed version of the query, so repeated questions get instant cached answers instead of hitting the LLM again.
- Rate limiting per client IP (via slowapi) to prevent abuse.
- Configurable retry logic throughout the pipeline (LLM calls, retrieval).

## Observability

- Structured JSON logging (every log line is a JSON object) — the kind of format that plugs directly into log aggregation tools like Datadog or ELK.
- Custom metrics collector tracking total requests, error rate, average latency, cache hit rate, and token usage, exposed via a `/metrics` endpoint.
- LangSmith tracing integrated throughout (via `@traceable` decorators) for full request-level tracing of the agent, security checks, and endpoint calls — useful for debugging what the agent actually did on a given request.
- `/health` endpoint for uptime checks / container orchestration (Docker, Kubernetes).

## API design

- Built with FastAPI, using Pydantic models for request/response validation.
- Modern FastAPI `lifespan` pattern for startup/shutdown (initializes the agent, cache, and security pipeline once at boot, not per-request).
- Endpoints: `POST /chat` (main RAG endpoint), `GET /health`, `GET /metrics`, `GET /cache/stats`.

## Deployment

- Configured for deployment on Render via `render.yaml` (infrastructure-as-code), with environment-based config (dev vs. production) and secrets kept out of source control.
- Environment/config management via Pydantic Settings, loading from a `.env` file.

## If asked "what was the hardest part" / debugging story

- Diagnosed a production auth bug where the LLM API calls were silently picking up a stale, invalid API key from the shell environment instead of the one in `.env`, because the LangChain client wasn't given the key explicitly and `python-dotenv` doesn't override existing environment variables by default. Fixed by passing the key explicitly to the client and forcing `.env` to take precedence. Good example of an environment-config bug that looked like a code bug at first.

---

### Honest gaps (good to know, not necessarily to volunteer)

- Cache and metrics are in-memory only — they reset on restart and wouldn't be shared across multiple server instances (would need Redis/Prometheus in a real multi-instance deployment).
- No conversation memory/history is persisted per thread yet (`thread_id` is passed but not used to load prior turns).
- `render.yaml` still references OpenAI env vars/models from an earlier iteration — worth updating to match the current Groq-based setup before deploying.
