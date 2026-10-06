import json
import logging

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_loop import _FakeAIMessage, _FakeLLM, _FakeTool, _settings

import agentic_rag.agent.loop as loop_module
import agentic_rag.observability.tracing as tracing_module
from agentic_rag.agent.loop import run_agent
from agentic_rag.observability.logging import JSONFormatter
from agentic_rag.observability.tracing import configure_tracing, get_tracer


@pytest.fixture
def exporter():
    exporter = InMemorySpanExporter()
    configure_tracing(exporter)
    yield exporter
    tracing_module._provider = None


def _record(message="hi"):
    return logging.LogRecord("l", logging.INFO, __file__, 1, message, (), None)


def test_agent_run_emits_nested_spans_for_steps_and_tools(exporter, monkeypatch):
    responses = [
        _FakeAIMessage(tool_calls=[{"name": "hybrid_retrieve", "args": {"query": "q"}, "id": "1"}]),
        _FakeAIMessage(
            tool_calls=[
                {"name": "generate_answer", "args": {"question": "q", "evidence": "e"}, "id": "2"}
            ]
        ),
    ]
    monkeypatch.setattr(loop_module, "get_llm", lambda model: _FakeLLM(responses))
    monkeypatch.setattr(
        loop_module,
        "build_tools",
        lambda **kw: [_FakeTool("hybrid_retrieve", "ev"), _FakeTool("generate_answer", "done")],
    )

    with get_tracer().start_as_current_span("POST /query"):
        run_agent("q", session=None, vector_store=None, settings=_settings())

    spans = {s.name: s for s in exporter.get_finished_spans()}
    names = [s.name for s in exporter.get_finished_spans()]
    assert names.count("agent.step") == 2
    assert {"POST /query", "agent.run", "tool.hybrid_retrieve", "tool.generate_answer"} <= set(spans)
    assert len({s.context.trace_id for s in spans.values()}) == 1
    assert spans["agent.run"].parent.span_id == spans["POST /query"].context.span_id
    assert spans["tool.hybrid_retrieve"].parent.span_id == spans["agent.run"].context.span_id


def test_log_lines_carry_trace_and_span_ids_inside_a_span(exporter):
    with get_tracer().start_as_current_span("work") as span:
        payload = json.loads(JSONFormatter().format(_record()))
        context = span.get_span_context()

    assert payload["trace_id"] == format(context.trace_id, "032x")
    assert payload["span_id"] == format(context.span_id, "016x")


def test_log_lines_have_no_trace_ids_outside_a_span():
    payload = json.loads(JSONFormatter().format(_record()))

    assert "trace_id" not in payload
