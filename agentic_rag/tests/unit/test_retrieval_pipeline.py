from langchain_core.documents import Document

import agentic_rag.retrieval.pipeline as retrieval_pipeline_module
from agentic_rag.config import Settings
from agentic_rag.retrieval.pipeline import retrieve


def _settings(**overrides) -> Settings:
    defaults = {
        "groq_api_key": "fake-groq",
        "pinecone_api_key": "fake-pinecone",
        "retrieval_k": 3,
        "hybrid_retrieval_enabled": False,
        "reranking_enabled": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _doc(source):
    return Document(page_content=f"content for {source}", metadata={"source": source})


class _FakeVectorStore:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def similarity_search(self, query, k):
        self.calls.append(k)
        return self._results[:k]


def test_plain_dense_fetches_exactly_retrieval_k_when_both_flags_off():
    vector_store = _FakeVectorStore([_doc(f"s{i}") for i in range(10)])

    results = retrieve(
        "q", session=None, vector_store=vector_store, settings=_settings(retrieval_k=3)
    )

    assert vector_store.calls == [3]
    assert len(results) == 3


def test_uses_hybrid_retrieve_instead_of_plain_dense_when_enabled(monkeypatch):
    calls = []

    def fake_hybrid_retrieve(session, query, *, vector_store, k):
        calls.append(k)
        return [_doc(f"h{i}") for i in range(k)]

    monkeypatch.setattr(retrieval_pipeline_module, "hybrid_retrieve", fake_hybrid_retrieve)
    vector_store = _FakeVectorStore([_doc("dense-only")])

    results = retrieve(
        "q",
        session=None,
        vector_store=vector_store,
        settings=_settings(retrieval_k=3, hybrid_retrieval_enabled=True),
    )

    assert vector_store.calls == []
    assert calls == [retrieval_pipeline_module.POOL_SIZE]
    assert len(results) == 3


def test_uses_larger_pool_for_plain_dense_when_reranking_enabled():
    vector_store = _FakeVectorStore([_doc(f"s{i}") for i in range(30)])

    retrieve(
        "q",
        session=None,
        vector_store=vector_store,
        settings=_settings(retrieval_k=3, reranking_enabled=True),
    )

    assert vector_store.calls == [retrieval_pipeline_module.POOL_SIZE]


def test_reranks_pool_and_truncates_to_retrieval_k(monkeypatch):
    reversed_order = []

    def fake_rerank(query, documents, *, model_name):
        reversed_order.append(model_name)
        return list(reversed(documents))

    monkeypatch.setattr(retrieval_pipeline_module, "rerank", fake_rerank)
    docs = [_doc(f"s{i}") for i in range(10)]
    vector_store = _FakeVectorStore(docs)

    results = retrieve(
        "q",
        session=None,
        vector_store=vector_store,
        settings=_settings(retrieval_k=2, reranking_enabled=True, reranker_model="fake-rr"),
    )

    assert reversed_order == ["fake-rr"]
    # docs[:POOL_SIZE] reversed, then truncated to retrieval_k=2
    expected = list(reversed(docs[: retrieval_pipeline_module.POOL_SIZE]))[:2]
    assert [d.metadata["source"] for d in results] == [d.metadata["source"] for d in expected]


def test_hybrid_and_rerank_compose(monkeypatch):
    def fake_hybrid_retrieve(session, query, *, vector_store, k):
        return [_doc(f"h{i}") for i in range(k)]

    def fake_rerank(query, documents, *, model_name):
        return list(reversed(documents))

    monkeypatch.setattr(retrieval_pipeline_module, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(retrieval_pipeline_module, "rerank", fake_rerank)
    vector_store = _FakeVectorStore([])

    results = retrieve(
        "q",
        session=None,
        vector_store=vector_store,
        settings=_settings(retrieval_k=2, hybrid_retrieval_enabled=True, reranking_enabled=True),
    )

    assert vector_store.calls == []
    assert len(results) == 2
