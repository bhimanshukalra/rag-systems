# Types of RAG (Retrieval-Augmented Generation)

There is no single canonical taxonomy of RAG systems. Most "types" are
overlapping design choices along a few independent dimensions (retrieval
strategy, retrieval mechanism, architecture, data modality) rather than
mutually exclusive categories — real systems usually combine several of
them (e.g. agentic + hybrid + reranked + contextual).

## Prominence and recommendations

Ranked by how commonly each is actually used in production, from most to
least prominent:

1. **Naive/Standard RAG** — the baseline every project starts with; not
   exciting but foundational.
2. **Hybrid retrieval (dense + sparse) + reranking** — the single
   highest-ROI upgrade over naive RAG; used in almost every production
   system.
3. **Agentic RAG** — the dominant direction right now; lets the model
   decide when/what to retrieve instead of blind top-k, handles multi-hop
   and tool use.
4. **Contextual Retrieval** — cheap, high-impact fix for the classic
   "chunk lost its context" problem; increasingly standard practice.
5. **Corrective RAG (CRAG)** — a practical reliability layer (detect bad
   retrieval, fall back to another source) rather than a novel
   architecture.
6. **GraphRAG** — matters specifically when data is relational/multi-hop
   (org charts, codebases, interconnected docs); overkill otherwise.

Less commonly built as standalone projects (more research/technique-level
than full architectures): RAPTOR, RETRO, HyDE, FLARE, Fusion RAG,
Speculative RAG.

**If picking one project to build:** an **agentic RAG system with hybrid
retrieval + reranking + contextual chunking** — that combination is what
most real deployments converge on. Add GraphRAG only if the target data is
genuinely relational.

## By retrieval strategy

- **Naive / Standard RAG** — single retrieval pass, then generate (retrieve
  top-k chunks, stuff into prompt).
- **Iterative / Multi-hop RAG** — retrieve, generate, then retrieve again
  based on intermediate results, for multi-step reasoning.
- **Recursive RAG** — retrieved results are used to refine/reformulate the
  query recursively.

## By when/how retrieval happens

- **Agentic RAG** — an LLM agent decides whether, when, and what to
  retrieve, and may call multiple tools/retrievers, rather than following a
  fixed retrieve-then-generate pipeline.
- **Adaptive RAG** — dynamically routes queries to different retrieval
  strategies (or skips retrieval entirely) based on query complexity.
- **Self-RAG** — the model critiques its own retrieved content and
  generations, deciding if retrieval was needed and reflecting on
  relevance/support.

## By retrieval mechanism

- **Dense retrieval RAG** — embedding-based vector similarity search.
- **Sparse retrieval RAG** — keyword-based (BM25/TF-IDF).
- **Hybrid RAG** — combines dense and sparse retrieval.
- **Graph RAG** — retrieves from a knowledge graph rather than (or
  alongside) a vector store, useful for multi-hop/relational queries.

## By architecture/structure

- **Fusion RAG (RAG-Fusion)** — generates multiple query variants, retrieves
  for each, then fuses/reranks the results.
- **Long-context RAG** — leans on large context windows with less
  aggressive chunking/retrieval.
- **Corrective RAG (CRAG)** — evaluates retrieved docs for relevance/
  correctness and falls back to web search or other sources if retrieval
  quality is poor.
- **Modular RAG** — a composable pipeline of swappable modules (retriever,
  reranker, generator, etc.).

## Query/retrieval refinement techniques

- **HyDE (Hypothetical Document Embeddings)** — generates a hypothetical
  answer first and embeds that to retrieve, instead of embedding the raw
  query.
- **FLARE (Forward-Looking Active Retrieval)** — predicts upcoming sentences
  and retrieves only when generation confidence drops.
- **Contextual Retrieval** — prepends chunk-specific context before
  embedding/indexing so isolated chunks retain document context.

## Structural/hierarchical variants

- **RAPTOR** — builds a tree of recursive summaries over the corpus and
  retrieves at multiple levels of abstraction.
- **RETRO** — retrieval is built into the model architecture/pretraining
  itself, rather than bolted on as a pipeline step.

## By data modality/source

- **Multimodal RAG** — retrieves across text, images, tables, audio.
- **Structured/SQL RAG** — retrieves from databases (often via text-to-SQL)
  rather than unstructured text.
- **Federated RAG** — queries multiple separate indices/sources and merges
  the results.

## Other angles

- **Conversational/Memory RAG** — incorporates dialogue history/long-term
  memory into retrieval.
- **Speculative RAG** — drafts multiple answers from different retrieved
  subsets in parallel, then verifies.
- **Temporal/Personalized RAG** — retrieval weighted by recency or
  user-specific context.
