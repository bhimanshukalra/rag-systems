from langchain_core.documents import Document

import agentic_rag.generation.synthesizer as synthesizer_module
from agentic_rag.generation.synthesizer import (
    NO_DOCUMENTS_ANSWER,
    generate_answer,
    get_llm,
)


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeRetryable:
    def __init__(self, response_content):
        self._response_content = response_content
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return _FakeLLMResponse(self._response_content)


class _FakeChatGroq:
    def __init__(self, *, model, temperature):
        self.model = model
        self.temperature = temperature
        self.retryable = _FakeRetryable("fake answer")
        self.stop_after_attempt = None

    def with_retry(self, *, stop_after_attempt):
        self.stop_after_attempt = stop_after_attempt
        return self.retryable


def _patch_chat_groq(monkeypatch):
    monkeypatch.setattr(synthesizer_module, "ChatGroq", _FakeChatGroq)
    get_llm.cache_clear()


def test_generate_answer_returns_llm_content_and_deduped_sources(monkeypatch):
    _patch_chat_groq(monkeypatch)
    docs = [
        Document(page_content="chunk one", metadata={"source": "https://a.test"}),
        Document(page_content="chunk two", metadata={"source": "https://a.test"}),
        Document(page_content="chunk three", metadata={"source": "https://b.test"}),
    ]

    result = generate_answer("What is X?", docs, model="fake-model")

    assert result.answer == "fake answer"
    assert result.sources == ["https://a.test", "https://b.test"]

    get_llm.cache_clear()


def test_generate_answer_includes_question_and_numbered_context(monkeypatch):
    _patch_chat_groq(monkeypatch)
    docs = [Document(page_content="the answer is 42", metadata={"source": "s1"})]

    generate_answer("What is X?", docs, model="fake-model")

    llm_instance = get_llm("fake-model")
    human_message = llm_instance.retryable.invoked_with[1]
    assert human_message[0] == "human"
    assert "What is X?" in human_message[1]
    assert "[1] (source: s1)" in human_message[1]
    assert "the answer is 42" in human_message[1]

    get_llm.cache_clear()


def test_generate_answer_uses_retry(monkeypatch):
    _patch_chat_groq(monkeypatch)
    docs = [Document(page_content="content", metadata={"source": "s1"})]

    generate_answer("q", docs, model="fake-model")

    assert get_llm("fake-model").stop_after_attempt == 3

    get_llm.cache_clear()


def test_generate_answer_returns_canned_response_with_no_documents(monkeypatch):
    calls = {"n": 0}

    class _ShouldNotBeCalled(_FakeChatGroq):
        def __init__(self, **kwargs):
            calls["n"] += 1
            super().__init__(**kwargs)

    monkeypatch.setattr(synthesizer_module, "ChatGroq", _ShouldNotBeCalled)
    get_llm.cache_clear()

    result = generate_answer("q", [], model="fake-model")

    assert result.answer == NO_DOCUMENTS_ANSWER
    assert result.sources == []
    assert calls["n"] == 0

    get_llm.cache_clear()


def test_get_llm_is_cached_per_model(monkeypatch):
    calls = {"n": 0}

    class _CountingFake(_FakeChatGroq):
        def __init__(self, **kwargs):
            calls["n"] += 1
            super().__init__(**kwargs)

    monkeypatch.setattr(synthesizer_module, "ChatGroq", _CountingFake)
    get_llm.cache_clear()

    get_llm("fake-model")
    get_llm("fake-model")

    assert calls["n"] == 1

    get_llm.cache_clear()
