When vector search fails:

1. Product codes
2. Error codes
3. Acronyms
4. Exact names

Vector vs BM25 search

vector search

1. good at semantic similarity, synonyms, natural questions
2. bad at exact matches, product codes and acronyms.

BM25

1. good at exact matches, rare terms, codes & IDs
2. bad at synonyms & semantic meaning.

What one misses the other catches

Hybrid is best of both

## Hybrid search pipeline

1. vector search and BM25 search are done independently
2. Results is merged via RRF (Reciprocal Rank Fusion) score [1/(k+rank)]
3. Documents good in both rise to the top

## Hybrid is ideal for:

(Hybrid rarely hurts, often helps)

1. Enterprise data with codes/IDs
2. Technical docs
3. Legal docs (statute numbers)
4. Mixed query types
5. Accuracy is critical

## Skip hybrid if

1. Simple Q&A chatbot
2. Creative writing assistant
3. Quick prototypes
4. Latency critical (adds ~20-50ms)

In production with real users - add hybrid search (the accuracy boost is worth it)

## Prod considerations for hybrid search

1. BM25 Rebuild - doesn't support incremental updates. It needs to be rebuilt when adding documents
2. Tune weights - start with 50/50. Adjust based on query patterns. Monitor which retriever contributes.
3. K value - Retrieve more, let RRF sort. k=4 or higher recommended
4. Latency - Hybrid add ~20-50 ms. Two searches instead of one. Worth it for accuracy.
