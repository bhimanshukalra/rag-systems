# agentic-rag

Agentic RAG service: multi-source ingestion, hybrid retrieval, and a
ReAct tool-calling agent behind a FastAPI API.

## Configuration

Copy `.env.example` to `.env` and fill in the required keys, including
`API_AUTH_TOKEN` (any long random string; every route except `/healthz`
requires it as `Authorization: Bearer <token>`).

## Run with Docker

```bash
docker compose up --build
curl http://localhost:8000/healthz
```

The SQLite registry and the Hugging Face model cache live in named volumes.
Pinecone stays external; there is no local vector DB container.

### Run exactly one worker

The image starts uvicorn with `--workers 1`, and that is a hard
constraint. The DB engine cache and the rate limiter's request counters
are in-process state. With several workers each gets its own counters, so
the effective rate limit multiplies by the worker count. Scale by
changing the limit, not the worker count.
