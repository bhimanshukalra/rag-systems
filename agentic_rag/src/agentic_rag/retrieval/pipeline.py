from langchain_core.documents import Document
from sqlmodel import Session

from agentic_rag.config import Settings
from agentic_rag.retrieval.hybrid import hybrid_retrieve
from agentic_rag.retrieval.reranker import rerank

# Candidate pool size when hybrid retrieval or reranking is enabled --
# both need more raw candidates to work with than the final answer uses.
POOL_SIZE = 20


def retrieve(
    question: str,
    *,
    session: Session | None,
    vector_store,
    settings: Settings,
) -> list[Document]:
    """Retrieve documents for a query, honoring the hybrid/rerank flags.

    Fetches a larger candidate pool only when hybrid retrieval or
    reranking is enabled; with both off, this fetches exactly the final
    k directly from dense retrieval -- identical to Phase 0's behavior,
    not just equivalent to it.
    """
    needs_pool = settings.hybrid_retrieval_enabled or settings.reranking_enabled
    pool_k = POOL_SIZE if needs_pool else settings.retrieval_k

    if settings.hybrid_retrieval_enabled:
        documents = hybrid_retrieve(session, question, vector_store=vector_store, k=pool_k)
    else:
        documents = vector_store.similarity_search(question, k=pool_k)

    if settings.reranking_enabled:
        documents = rerank(question, documents, model_name=settings.reranker_model)

    return documents[: settings.retrieval_k]
