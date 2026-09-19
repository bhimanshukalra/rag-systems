# rag-systems

A collection of RAG architectures explored hands-on — naive RAG, agentic RAG, graph RAG, and more — each as a self-contained project.

## Projects

- [`agentic_rag/`](agentic_rag/) — an agentic RAG assistant built with LangGraph: routes questions to a private Pinecone-backed knowledge base or a direct answer, grades retrieved evidence, rewrites the query and retries once if it's weak, and falls back to Tavily web search.

Each project has its own `pyproject.toml`, dependencies, and README — `cd` into one and follow its own setup instructions.
