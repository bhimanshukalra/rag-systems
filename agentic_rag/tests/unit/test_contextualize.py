from langchain_core.documents import Document

import agentic_rag.generation.synthesizer as synthesizer_module
from agentic_rag.generation.synthesizer import get_llm
from agentic_rag.ingestion.contextualize import (
    contextualize_chunk,
    contextualize_chunks,
)


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeRetryable:
    def __init__(self):
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return _FakeResponse("  This chunk covers widget pricing.  ")


class _FakeChatGroq:
    def __init__(self, *, model, temperature):
        self.retryable = _FakeRetryable()

    def with_retry(self, *, stop_after_attempt):
        return self.retryable


def _patch_llm(monkeypatch):
    monkeypatch.setattr(synthesizer_module, "ChatGroq", _FakeChatGroq)
    get_llm.cache_clear()


def test_contextualize_chunk_prepends_stripped_blurb(monkeypatch):
    _patch_llm(monkeypatch)

    result = contextualize_chunk("full document text", "the original chunk", model="fake-model")

    assert result == "This chunk covers widget pricing.\n\nthe original chunk"

    get_llm.cache_clear()


def test_contextualize_chunk_caps_document_context_length(monkeypatch):
    _patch_llm(monkeypatch)
    long_document = "x" * 10_000

    contextualize_chunk(long_document, "chunk", model="fake-model")

    llm = get_llm("fake-model")
    human_message = llm.retryable.invoked_with[1][1]
    excerpt_length = len(human_message) - len("Document excerpt:\n\n\nChunk:\nchunk")
    assert excerpt_length == 1500

    get_llm.cache_clear()


def test_contextualize_chunks_preserves_metadata_and_count(monkeypatch):
    _patch_llm(monkeypatch)
    chunks = [
        Document(page_content="chunk one", metadata={"source": "a", "start_index": 0}),
        Document(page_content="chunk two", metadata={"source": "a", "start_index": 10}),
    ]

    results = contextualize_chunks(chunks, "document text", model="fake-model")

    assert len(results) == 2
    assert results[0].metadata == {"source": "a", "start_index": 0}
    assert "chunk one" in results[0].page_content
    assert "chunk two" in results[1].page_content

    get_llm.cache_clear()
