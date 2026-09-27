from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, documents: list[Document], *, model_name: str) -> list[Document]:
    """Re-score and re-sort `documents` by cross-encoder relevance to `query`.

    Judges each candidate directly against the query, rather than relying
    on cross-retriever rank agreement (RRF's approach) -- this is what
    catches a relevant document whose chunks got split across dense and
    sparse retrieval without either single chunk being favored by both.
    Returns the same documents, reordered; never drops any.
    """
    if not documents:
        return []

    reranker = get_reranker(model_name)
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
    return [doc for _score, doc in ranked]
