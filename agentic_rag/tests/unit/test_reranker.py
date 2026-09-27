from langchain_core.documents import Document

import agentic_rag.retrieval.reranker as reranker_module
from agentic_rag.retrieval.reranker import get_reranker, rerank


class _FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.predict_calls = []

    def predict(self, pairs):
        self.predict_calls.append(pairs)
        # Score by how many times "relevant" appears in the doc text --
        # deterministic and easy to reason about in assertions.
        return [doc_text.count("relevant") for _query, doc_text in pairs]


def _patch_cross_encoder(monkeypatch):
    monkeypatch.setattr(reranker_module, "CrossEncoder", _FakeCrossEncoder)
    get_reranker.cache_clear()


def test_rerank_sorts_by_descending_score(monkeypatch):
    _patch_cross_encoder(monkeypatch)
    docs = [
        Document(page_content="nothing here at all", metadata={"source": "low"}),
        Document(page_content="relevant relevant relevant", metadata={"source": "high"}),
        Document(page_content="somewhat relevant text", metadata={"source": "mid"}),
    ]

    results = rerank("query", docs, model_name="fake-model")

    assert [doc.metadata["source"] for doc in results] == ["high", "mid", "low"]

    get_reranker.cache_clear()


def test_rerank_preserves_all_documents(monkeypatch):
    _patch_cross_encoder(monkeypatch)
    docs = [Document(page_content=f"doc {i}", metadata={"source": f"s{i}"}) for i in range(5)]

    results = rerank("query", docs, model_name="fake-model")

    assert {doc.metadata["source"] for doc in results} == {doc.metadata["source"] for doc in docs}
    assert len(results) == 5

    get_reranker.cache_clear()


def test_rerank_builds_query_document_pairs_in_order(monkeypatch):
    _patch_cross_encoder(monkeypatch)
    docs = [
        Document(page_content="alpha", metadata={"source": "a"}),
        Document(page_content="beta", metadata={"source": "b"}),
    ]

    rerank("what is it", docs, model_name="fake-model")

    encoder = get_reranker("fake-model")
    assert encoder.predict_calls[-1] == [("what is it", "alpha"), ("what is it", "beta")]

    get_reranker.cache_clear()


def test_rerank_handles_empty_input(monkeypatch):
    _patch_cross_encoder(monkeypatch)

    assert rerank("query", [], model_name="fake-model") == []

    get_reranker.cache_clear()


def test_get_reranker_is_cached_per_model(monkeypatch):
    calls = {"n": 0}

    class _CountingFake(_FakeCrossEncoder):
        def __init__(self, model_name):
            calls["n"] += 1
            super().__init__(model_name)

    monkeypatch.setattr(reranker_module, "CrossEncoder", _CountingFake)
    get_reranker.cache_clear()

    get_reranker("fake-model")
    get_reranker("fake-model")

    assert calls["n"] == 1

    get_reranker.cache_clear()
