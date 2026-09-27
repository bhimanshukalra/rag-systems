from typing import ClassVar

from langchain_core.documents import Document

import agentic_rag.agent.tools as tools_module
from agentic_rag.agent.grading import EvidenceGrade
from agentic_rag.agent.tools import build_tools
from agentic_rag.config import Settings


def _settings(**overrides) -> Settings:
    defaults = {
        "groq_api_key": "fake-groq",
        "pinecone_api_key": "fake-pinecone",
        "llm_model": "fake-llm",
        "tavily_api_key": "fake-tavily",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


def test_build_tools_returns_five_tools_with_expected_names():
    tools = build_tools(session=None, vector_store=None, settings=_settings())

    names = {t.name for t in tools}
    assert names == {
        "hybrid_retrieve",
        "grade_evidence",
        "web_search",
        "rewrite_query",
        "generate_answer",
    }


def test_hybrid_retrieve_tool_formats_documents(monkeypatch):
    captured = {}

    def fake_retrieve(query, *, session, vector_store, settings):
        captured.update(query=query, session=session, vector_store=vector_store)
        return [Document(page_content="chunk text", metadata={"source": "s1"})]

    monkeypatch.setattr(tools_module, "retrieve", fake_retrieve)

    tools = build_tools(session="sess", vector_store="vs", settings=_settings())
    result = _tool(tools, "hybrid_retrieve").invoke({"query": "widget warranty"})

    assert "[1] (source: s1)" in result
    assert "chunk text" in result
    assert captured == {"query": "widget warranty", "session": "sess", "vector_store": "vs"}


def test_hybrid_retrieve_tool_handles_no_results(monkeypatch):
    monkeypatch.setattr(tools_module, "retrieve", lambda *a, **k: [])

    tools = build_tools(session=None, vector_store=None, settings=_settings())
    result = _tool(tools, "hybrid_retrieve").invoke({"query": "q"})

    assert result == "No results found."


def test_grade_evidence_tool_formats_grade(monkeypatch):
    captured = {}

    def fake_grade_evidence(question, evidence, *, model):
        captured.update(question=question, evidence=evidence, model=model)
        return EvidenceGrade(sufficient=False, reason="missing pricing info")

    monkeypatch.setattr(tools_module, "grade_evidence", fake_grade_evidence)

    tools = build_tools(session=None, vector_store=None, settings=_settings(llm_model="grader-model"))
    result = _tool(tools, "grade_evidence").invoke({"question": "q?", "evidence": "e"})

    assert "sufficient=False" in result
    assert "missing pricing info" in result
    assert captured == {"question": "q?", "evidence": "e", "model": "grader-model"}


def test_web_search_tool_delegates_with_api_key(monkeypatch):
    calls = []

    def fake_search_web(query, *, api_key):
        calls.append((query, api_key))
        return "web results text"

    monkeypatch.setattr(tools_module, "search_web", fake_search_web)

    tools = build_tools(
        session=None, vector_store=None, settings=_settings(tavily_api_key="tavily-key")
    )
    result = _tool(tools, "web_search").invoke({"query": "weather today"})

    assert result == "web results text"
    assert calls == [("weather today", "tavily-key")]


def test_rewrite_query_tool_calls_llm_and_strips_content(monkeypatch):
    class _FakeResponse:
        content = "  improved query text  "

    class _FakeRetryable:
        def invoke(self, messages):
            self.invoked_with = messages
            return _FakeResponse()

    class _FakeLLM:
        def with_retry(self, *, stop_after_attempt):
            self.stop_after_attempt = stop_after_attempt
            self.retryable = _FakeRetryable()
            return self.retryable

    fake_llm = _FakeLLM()
    monkeypatch.setattr(tools_module, "get_llm", lambda model: fake_llm)

    tools = build_tools(session=None, vector_store=None, settings=_settings())
    result = _tool(tools, "rewrite_query").invoke(
        {"original_query": "old query", "reason": "too vague"}
    )

    assert result == "improved query text"
    assert fake_llm.stop_after_attempt == 3
    human_message = fake_llm.retryable.invoked_with[1]
    assert "old query" in human_message[1]
    assert "too vague" in human_message[1]


def test_generate_answer_tool_wraps_evidence_and_returns_answer_text(monkeypatch):
    captured = {}

    class _FakeResult:
        answer = "final answer text"
        sources: ClassVar[list] = ["agent-gathered evidence"]

    def fake_generate_answer(question, documents, *, model):
        captured.update(question=question, documents=documents, model=model)
        return _FakeResult()

    monkeypatch.setattr(tools_module, "generate_answer", fake_generate_answer)

    tools = build_tools(session=None, vector_store=None, settings=_settings(llm_model="answer-model"))
    result = _tool(tools, "generate_answer").invoke(
        {"question": "q?", "evidence": "gathered evidence text"}
    )

    assert result == "final answer text"
    assert captured["question"] == "q?"
    assert captured["model"] == "answer-model"
    assert len(captured["documents"]) == 1
    assert captured["documents"][0].page_content == "gathered evidence text"
    assert captured["documents"][0].metadata["source"] == "agent-gathered evidence"
