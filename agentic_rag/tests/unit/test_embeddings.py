import agentic_rag.indexing.embeddings as embeddings_module
from agentic_rag.indexing.embeddings import get_embeddings


class _FakeHuggingFaceEmbeddings:
    def __init__(self, *, model_name, encode_kwargs):
        self.model_name = model_name
        self.encode_kwargs = encode_kwargs


def test_passes_model_name_and_normalizes(monkeypatch):
    monkeypatch.setattr(
        embeddings_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings
    )
    get_embeddings.cache_clear()

    result = get_embeddings("some/model")

    assert result.model_name == "some/model"
    assert result.encode_kwargs == {"normalize_embeddings": True}

    get_embeddings.cache_clear()


def test_is_cached_per_model_name(monkeypatch):
    call_count = {"n": 0}

    class _CountingFake(_FakeHuggingFaceEmbeddings):
        def __init__(self, **kwargs):
            call_count["n"] += 1
            super().__init__(**kwargs)

    monkeypatch.setattr(embeddings_module, "HuggingFaceEmbeddings", _CountingFake)
    get_embeddings.cache_clear()

    get_embeddings("some/model")
    get_embeddings("some/model")

    assert call_count["n"] == 1

    get_embeddings.cache_clear()
