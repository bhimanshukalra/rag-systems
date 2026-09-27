from langchain_core.documents import Document
from sqlmodel import Session

from agentic_rag.indexing import keyword_index
from agentic_rag.indexing.vector_store import get_chunk_id

RRF_K = 60
DEFAULT_POOL_SIZE = 20


def hybrid_retrieve(
    session: Session,
    query: str,
    *,
    vector_store,
    k: int,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> list[Document]:
    """Fuse dense (Pinecone) and sparse (BM25) retrieval via Reciprocal
    Rank Fusion (RRF).

    Each ranked list contributes 1/(RRF_K + rank) per chunk (1-indexed
    rank within that list); a chunk present in both lists sums both
    contributions, so something both retrievers agree on outranks
    something only one of them liked -- even if that one ranked it 1st.
    Chunk identity is get_chunk_id() (hash of source + position +
    content), which both indexes were populated with, so the same
    underlying chunk fuses into one entry regardless of which retriever
    returned it.
    """
    dense_results = vector_store.similarity_search(query, k=pool_size)
    sparse_results = keyword_index.search(session, query, k=pool_size)

    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Document] = {}

    for ranked_list in (dense_results, sparse_results):
        for rank, doc in enumerate(ranked_list, start=1):
            chunk_id = get_chunk_id(doc)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
            chunks_by_id.setdefault(chunk_id, doc)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks_by_id[chunk_id] for chunk_id in ranked_ids[:k]]
