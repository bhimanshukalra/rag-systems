# Code review: agentic_rag

Scope: all of `src/` plus the Docker setup. This is a read-through only. The tests and live Pinecone/Groq calls were not run. Finding 3 depends on how Pinecone returns numbers and needs one live check.

## Major

### 1. Any API user can read arbitrary local files, then get them back through `/query`
Files: `src/agentic_rag/api/schemas.py:9`, `src/agentic_rag/ingestion/pipeline.py:19`, `src/agentic_rag/ingestion/loaders/markdown.py`

`POST /sources` accepts `source_type: "markdown"` with any `location`. `load_markdown` calls `Path(location).read_text()` with no restriction. A caller can send `{"source_type":"markdown","location":"/app/.env"}`. That ingests the Groq, Pinecone and auth keys into the index, and a later `/query` returns them. The URL safety checks only cover `web` and `pdf`.

Fix: make the local-file loader CLI-only, or limit it to a fixed directory and check the resolved path.

### 2. Timed-out queries keep running, use a closed DB session, and fill the thread pool
Files: `src/agentic_rag/api/routes.py:30`, `src/agentic_rag/api/routes.py:101`

- `future.cancel()` does nothing once the thread has started. After the 504, the request-scoped `Session` from `get_session` is closed while the worker thread still uses it for hybrid retrieval. SQLite is opened with `check_same_thread=False`, so this fails silently.
- The pool has 4 workers and nothing bounds the queue. A few slow or hung Groq calls leave every later `/query` waiting in the queue until its 60 s timeout. Time spent queued counts against that timeout.

Fix: build a fresh session inside the worker, bound the queue, and put a deadline check inside the agent loop.

### 3. Hybrid fusion probably never merges dense and sparse hits
Files: `src/agentic_rag/retrieval/hybrid.py:36`, `src/agentic_rag/indexing/vector_store.py:14`

The chunk ID hashes `metadata["start_index"]`. In SQLite it is stored as JSON, so it round-trips as an int and hashes as `"1234"`. Pinecone returns numeric metadata as floats, so it would hash as `"1234.0"`. If so, the same chunk gets two different IDs. RRF then treats each result as a separate chunk, and the "both retrievers agree" boost never fires. The hybrid eval gains could come from interleaving rather than fusion.

Check: print `get_chunk_id` for a dense hit and the matching BM25 hit.

Fix: normalise `int(start_index)` inside `get_chunk_id`, or look chunks up by the stored vector ID. Changing the hash means re-ingesting existing data.

### 4. The agent has to retype its evidence
Files: `src/agentic_rag/agent/tools.py:46`, `src/agentic_rag/agent/tools.py:114`

`grade_evidence` and `generate_answer` take the evidence as an LLM-written argument. The model must copy every retrieved chunk into the tool call, up to twice per question. That costs tokens, hits output limits, and invites the malformed-tool-call 400s already special-cased in `loop.py`. It also lets the model paraphrase or invent "evidence". The final answer is generated from the model's own summary rather than the retrieved text.

- All citations point to one synthetic document, `agent-gathered evidence`.
- `QueryResponse.sources` is always `[]`.

Fix: keep retrieved documents in loop state, and have the tools take no evidence or only chunk IDs.

## Moderate

### 5. `/sources` runs ingestion inside the request and returns 201 even when it failed
File: `src/agentic_rag/api/routes.py:51`

- Ingestion runs synchronously with no overall timeout. A large PDF, or contextual chunking (one LLM call per chunk), can hold a request for minutes.
- `safe_get` does not cap response size, and the `requests` timeout is per read, not total. A big or slow response can exhaust memory or hold a worker.
- A failure comes back as 201 with `error_message=str(exc)`, which leaks internal errors. The "resolves to disallowed address X" messages also let a caller probe internal DNS.
- Two concurrent requests for the same new source race on the unique `content_hash`, and one gets a 500. A crash mid-ingest leaves the record stuck in `in_progress`.

### 6. Re-ingestion never refreshes data, and stale chunks are never removed
File: `src/agentic_rag/ingestion/pipeline.py:30`

Once a source is `ready`, it is never re-fetched. There is no delete or update endpoint, so outdated chunks stay in Pinecone and BM25 permanently. With contextual chunking, the blurb is part of the chunk content and therefore part of the ID. If the LLM output varies, re-running creates duplicate vectors.

### 7. `/query` returns a 500 when the Pinecone index doesn't exist
File: `src/agentic_rag/indexing/vector_store.py:77`

On a fresh deploy with nothing ingested, `get_vector_store` calls `pinecone.Index(name)`, and the query path fails with an unhandled error. A cold embedding-model download can also exceed the 60 s timeout.

### 8. The agent can loop uselessly and has a prompt-injection surface
File: `src/agentic_rag/agent/loop.py:58`

- If a tool keeps erroring, or the model keeps re-grading, the loop spends all 6 steps. Each step sends the full message history, including the large evidence blobs.
- `_force_final_answer` concatenates every `ToolMessage` with no size cap, which can exceed the model's context window.
- Web and PDF content is fed to the model unfiltered. The tools are read-only, so the damage is limited to manipulated answers.

## Minor

- **Rate limiting:** `api/rate_limit.py` uses one global counter shared by all callers. A single client can lock everyone else out with 30 requests. Fixed windows also allow a 2x burst at the window edge.
- **BM25:** `indexing/keyword_index.py:76` rebuilds the BM25 index and loads every row on each retrieve call. The docstring accepts this, but it repeats on every agent step.
- **Auth token:** `API_AUTH_TOKEN` has no minimum length. An empty value makes an empty bearer header authenticate.
- **Typing and empty-answer handling:**
  - `/query` takes any string length with no cap, which matters for Groq token cost.
  - `run_agent` returns `response.content`, which can be a list for some models even though the type is `str`.
  - `NO_DOCUMENTS_ANSWER` is unreachable from the agent path, because `generate_answer` always wraps a non-empty synthetic document.
- **DNS rebinding:** `safe_get` validates DNS and then connects with a second lookup. The code documents this as accepted, which is reasonable.
- **Tracing:** the full `question` is set as a span attribute and exported. Check that this is acceptable for the telemetry backend.

## What's solid

- The URL safety design checks every redirect hop and every resolved IP.
- The auth comparison is constant-time.
- Chunk IDs are deterministic, which makes upserts idempotent, apart from the float issue in finding 3.
- The Docker image is non-root, runs one worker (documented), and bakes in no secrets.
- `tests/` and `planning/` are excluded from the build context.

## Suggested order

1. Lock down or remove the `markdown` source type over the API (finding 1).
2. Verify and fix the chunk-ID float mismatch (finding 3).
3. Stop passing evidence through LLM arguments, which fixes finding 4 and brings real citations back.
4. Give the worker thread its own DB session and a bounded queue (finding 2).
5. Make ingestion async or timeout-bounded, with a response size cap (finding 5).
