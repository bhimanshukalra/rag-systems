from langchain_core.documents import Document

from agentic_rag.retrieval import hybrid as hybrid_module
from agentic_rag.retrieval.hybrid import hybrid_retrieve


def _doc(source, content="content"):
    return Document(page_content=content, metadata={"source": source})


class _FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def similarity_search(self, query, k):
        return self._results[:k]


def _patch_keyword_search(monkeypatch, results):
    monkeypatch.setattr(hybrid_module.keyword_index, "search", lambda session, query, k: results[:k])


def test_document_ranked_first_in_both_lists_wins(monkeypatch):
    doc_a = _doc("a")
    doc_b = _doc("b")
    doc_c = _doc("c")

    vector_store = _FakeVectorStore([doc_a, doc_b, doc_c])
    _patch_keyword_search(monkeypatch, [doc_a, doc_c, doc_b])

    results = hybrid_retrieve(None, "q", vector_store=vector_store, k=3)

    assert results[0].metadata["source"] == "a"


def test_includes_documents_present_in_only_one_retriever(monkeypatch):
    doc_a = _doc("a")
    doc_dense_only = _doc("dense-only")

    vector_store = _FakeVectorStore([doc_a, doc_dense_only])
    _patch_keyword_search(monkeypatch, [doc_a])

    results = hybrid_retrieve(None, "q", vector_store=vector_store, k=5)

    sources = {doc.metadata["source"] for doc in results}
    assert "dense-only" in sources


def test_deduplicates_same_chunk_from_both_retrievers(monkeypatch):
    # Same page_content + metadata -> same get_chunk_id(), even though
    # they're different Document object instances.
    doc_dense = Document(page_content="shared", metadata={"source": "a"})
    doc_sparse = Document(page_content="shared", metadata={"source": "a"})

    vector_store = _FakeVectorStore([doc_dense])
    _patch_keyword_search(monkeypatch, [doc_sparse])

    results = hybrid_retrieve(None, "q", vector_store=vector_store, k=5)

    assert len(results) == 1


def test_agreement_across_retrievers_beats_a_single_top_rank(monkeypatch):
    # doc_agreed is ranked 5th by BOTH retrievers; doc_solo is ranked 1st
    # by dense only. RRF's summed contribution for agreement should still
    # win: 1/65 + 1/65 (~0.0308) > 1/61 (~0.0164).
    doc_agreed = _doc("agreed")
    doc_solo = _doc("solo")
    filler_dense = [_doc(f"dense-filler-{i}") for i in range(4)]
    filler_sparse = [_doc(f"sparse-filler-{i}") for i in range(4)]

    vector_store = _FakeVectorStore([doc_solo, *filler_dense, doc_agreed])
    _patch_keyword_search(monkeypatch, [*filler_sparse, doc_agreed])

    results = hybrid_retrieve(None, "q", vector_store=vector_store, k=1, pool_size=10)

    assert results[0].metadata["source"] == "agreed"


def test_respects_k(monkeypatch):
    docs = [_doc(f"s{i}") for i in range(10)]
    vector_store = _FakeVectorStore(docs)
    _patch_keyword_search(monkeypatch, [])

    results = hybrid_retrieve(None, "q", vector_store=vector_store, k=3, pool_size=10)

    assert len(results) == 3


def test_pool_size_limits_raw_fetch_from_each_retriever(monkeypatch):
    calls = {}

    class TrackingVectorStore:
        def similarity_search(self, query, k):
            calls["dense_k"] = k
            return []

    def fake_search(session, query, k):
        calls["sparse_k"] = k
        return []

    monkeypatch.setattr(hybrid_module.keyword_index, "search", fake_search)

    hybrid_retrieve(None, "q", vector_store=TrackingVectorStore(), k=5, pool_size=7)

    assert calls["dense_k"] == 7
    assert calls["sparse_k"] == 7
