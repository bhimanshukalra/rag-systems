# agentic-rag

An agentic RAG assistant built with LangGraph. Given a question, it routes to a
private knowledge base (a single scraped doc page, chunked and embedded into
Pinecone) or a direct answer, grades whether the retrieved evidence is good
enough, falls back to Tavily web search when it isn't, and rewrites the query
for one retry before giving up.

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with:

- `GROQ_API_KEY` — the LLM used for routing, grading, rewriting, and answering
- `TAVILY_API_KEY` — web search fallback
- `PINECONE_API_KEY` — the vector store

## Usage

The first run needs `--reingest` to scrape, chunk, embed, and upload the
source into Pinecone before there's anything to query:

```bash
uv run python src/main.py --reingest "What is Agentic RAG?"
```

After that, ingestion is skipped by default — just ask a question against the
already-populated index:

```bash
uv run python src/main.py "How does query rewriting work?"
```

## Configuration

These are optional overrides (not in `.env.example` since they're not
required) — defaults ingest a single LangGraph docs page:

- `INGEST_SOURCE_URL` — the page to scrape
- `PINECONE_INDEX_NAME` — the Pinecone index to create/use
- `PINECONE_NAMESPACE` — the namespace within that index

## Tests

```bash
uv run pytest
```
