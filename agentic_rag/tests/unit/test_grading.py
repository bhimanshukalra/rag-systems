from typing import ClassVar

import agentic_rag.generation.synthesizer as synthesizer_module
from agentic_rag.agent.grading import EvidenceGrade, grade_evidence
from agentic_rag.generation.synthesizer import get_llm


class _FakeStructuredLLM:
    def __init__(self, result):
        self.result = result
        self.invoked_with = None
        self.stop_after_attempt = None

    def with_retry(self, *, stop_after_attempt):
        self.stop_after_attempt = stop_after_attempt
        return self

    def invoke(self, messages):
        self.invoked_with = messages
        return self.result


class _FakeChatGroq:
    default_result: ClassVar[EvidenceGrade | None] = None

    def __init__(self, *, model, temperature):
        self.model = model
        self.temperature = temperature
        self.structured_llm = None

    def with_structured_output(self, schema):
        result = _FakeChatGroq.default_result or schema(
            sufficient=True, reason="looks sufficient"
        )
        self.structured_llm = _FakeStructuredLLM(result)
        return self.structured_llm


def _patch_llm(monkeypatch):
    monkeypatch.setattr(synthesizer_module, "ChatGroq", _FakeChatGroq)
    get_llm.cache_clear()


def test_grade_evidence_returns_structured_result(monkeypatch):
    _patch_llm(monkeypatch)

    result = grade_evidence("q", "evidence text", model="fake-model")

    assert isinstance(result, EvidenceGrade)
    assert result.sufficient is True

    get_llm.cache_clear()


def test_grade_evidence_reports_insufficient_with_reason(monkeypatch):
    _patch_llm(monkeypatch)
    _FakeChatGroq.default_result = EvidenceGrade(
        sufficient=False, reason="missing pricing info"
    )

    result = grade_evidence("q", "evidence text", model="fake-model")

    assert result.sufficient is False
    assert result.reason == "missing pricing info"

    _FakeChatGroq.default_result = None
    get_llm.cache_clear()


def test_grade_evidence_sends_question_and_evidence(monkeypatch):
    _patch_llm(monkeypatch)

    grade_evidence("what is X?", "some evidence", model="fake-model")

    llm_instance = get_llm("fake-model")
    human_message = llm_instance.structured_llm.invoked_with[1]
    assert human_message[0] == "human"
    assert "what is X?" in human_message[1]
    assert "some evidence" in human_message[1]

    get_llm.cache_clear()


def test_grade_evidence_uses_retry(monkeypatch):
    _patch_llm(monkeypatch)

    grade_evidence("q", "e", model="fake-model")

    llm_instance = get_llm("fake-model")
    assert llm_instance.structured_llm.stop_after_attempt == 3

    get_llm.cache_clear()
