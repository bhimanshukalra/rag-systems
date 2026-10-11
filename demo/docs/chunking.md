## Chunks

Each chunk is embedded in isolation. The model only sees fragments and not the complete concept, so ensuring that embedding captures complete meaning is essential.

Below-mentioned variables affect chunking quality.

1. Chunk size: the sweet spot is between 200 and 1,000 tokens. Too large, and it dilutes the meaning. Too small, and it loses context.
2. Overlap: A good overlap is between 10% and 20%, which preserves context.
3. Split boundaries:
   - Fixed means random cuts.
   - Recursive is cutting at the end of paragraphs or sentences.
   - Semantic is ensuring that we are cutting at meaning boundary.
4. Content type: depending on whether the documents are code, legal documents or markdown different treatment is required.

## Fixed size chunking

cut at exact intervals e.g. every 500 chars

Pros:

1. Simple implementation
2. Fast processing
3. Predictable sizes

Cons:

1. High chances of destroying meaning
2. Poor quality results
3. Inaccurate retrieval
4. Can lead to frustrating UX

Not for Prod

## Recursive chunking

Start by splitting on paragraphs. If it's still too big:

- Split on new lines.
- If still too big, split on sentences.
- If still too big, split on clauses.
- If still too big, split on words.
- As a last resort, split on characters.

This is a good balance of speed and quality.
The recommended default.

## Semantic chunking

Embed each sentence, compare adjacent embeddings, and split when similarity drops.
In this chunking strategy, we split at meaning boundaries, and we try to chunk the embeddings based on the closeness of similarity score.

## Late chunking

In traditional chunking, chunk 5 has no idea what chunks 1 to 4 contain. In lead chunking, chunk 5's embedding includes full document context. This results in 10-12 push attack case improvements. Full document context is preserved.

## Chunking decision framework

- If we have to prototype quickly, there is recursive.
- If the document is simple and structured, then we go for recursive.
- If the quality is critical, we go with semantic.
- If it's complex and topic-shifting, we go with semantic.
- If none of the above is true, then we go with recursive.

It's the 80/20 rule: the recursive will take you 80% of the way. The last 20% would be semantic.

# Contextual chunking

Contextual chunking (often tied to Contextual Retrieval or Contextual Embeddings) is a data preprocessing technique for Retrieval-Augmented Generation (RAG) systems that attaches a short, LLM-generated description or prefix to each document chunk before it is embedded and indexed.

# Late chunking vs early chunking

Early chunking (traditional chunking) splits text into smaller pieces before generating embeddings, whereas late chunking passes the entire document through a long-context embedding model first and chunks the token-level embeddings afterward.

Comparison Breakdown

Feature | Early Chunking (Traditional) | Late Chunking
Order of Operations | Split raw text into chunks → Embed each chunk independently. | Embed the full document first (using a long-context model) → Split token embeddings and pool.

Context Awareness | Low to moderate. Chunks lose global document context and reference links across boundaries. | High. Every chunk vector carries full-document and neighboring token semantics.

Compute & Latency | Low latency, fast ingestion, and minimal compute overhead. | Higher latency and compute intensity due to full-document transformer passes.

Model Requirements | Works with standard, short-context embedding models. | Requires a long-context embedding model (e.g., supporting 8K+ tokens like Jina Embeddings v2).

Best Used For | Static knowledge bases, simple text structures, strict low-latency SLAs, and tight budgets. | Complex legal/medical texts, cross-chunk reference heavy documents, and high-accuracy RAG.
