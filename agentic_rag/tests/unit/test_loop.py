import agentic_rag.agent.loop as loop_module
from agentic_rag.agent.loop import run_agent
from agentic_rag.config import Settings


def _settings(**overrides) -> Settings:
    defaults = {
        "groq_api_key": "fake-groq",
        "pinecone_api_key": "fake-pinecone",
        "llm_model": "fake-llm",
        "agent_max_steps": 6,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class _FakeAIMessage:
    def __init__(self, *, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeRetryable:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.invoked_messages = []

    def invoke(self, messages):
        self.invoked_messages.append(list(messages))
        return next(self._responses)


class _FakeBoundLLM:
    def __init__(self, responses):
        self.retryable = _FakeRetryable(responses)
        self.bound_tools = None

    def with_retry(self, *, stop_after_attempt):
        return self.retryable


class _FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.last_bound = None

    def bind_tools(self, tools):
        self.last_bound = _FakeBoundLLM(self.responses)
        self.last_bound.bound_tools = tools
        return self.last_bound


class _FakeTool:
    def __init__(self, name, result_or_fn):
        self.name = name
        self._result_or_fn = result_or_fn
        self.invoked_with = []

    def invoke(self, args):
        self.invoked_with.append(args)
        if callable(self._result_or_fn):
            return self._result_or_fn(args)
        return self._result_or_fn


def test_returns_direct_answer_when_no_tool_calls(monkeypatch):
    fake_llm = _FakeLLM([_FakeAIMessage(content="direct answer, no tools needed")])
    monkeypatch.setattr(loop_module, "get_llm", lambda model: fake_llm)
    monkeypatch.setattr(loop_module, "build_tools", lambda **kwargs: [])

    result = run_agent("q", session=None, vector_store=None, settings=_settings())

    assert result == "direct answer, no tools needed"


def test_calls_tools_and_returns_on_generate_answer(monkeypatch):
    responses = [
        _FakeAIMessage(
            tool_calls=[{"name": "hybrid_retrieve", "args": {"query": "q"}, "id": "1"}]
        ),
        _FakeAIMessage(
            tool_calls=[
                {"name": "generate_answer", "args": {"question": "q", "evidence": "e"}, "id": "2"}
            ]
        ),
    ]
    fake_llm = _FakeLLM(responses)
    monkeypatch.setattr(loop_module, "get_llm", lambda model: fake_llm)
    fake_tools = [
        _FakeTool("hybrid_retrieve", "retrieved evidence text"),
        _FakeTool("generate_answer", "final answer"),
    ]
    monkeypatch.setattr(loop_module, "build_tools", lambda **kwargs: fake_tools)

    result = run_agent("q", session=None, vector_store=None, settings=_settings())

    assert result == "final answer"


def test_supports_multiple_retrieve_calls_before_finishing(monkeypatch):
    responses = [
        _FakeAIMessage(
            tool_calls=[{"name": "hybrid_retrieve", "args": {"query": "topic A"}, "id": "1"}]
        ),
        _FakeAIMessage(
            tool_calls=[{"name": "hybrid_retrieve", "args": {"query": "topic B"}, "id": "2"}]
        ),
        _FakeAIMessage(
            tool_calls=[
                {
                    "name": "generate_answer",
                    "args": {"question": "q", "evidence": "combined"},
                    "id": "3",
                }
            ]
        ),
    ]
    fake_llm = _FakeLLM(responses)
    monkeypatch.setattr(loop_module, "get_llm", lambda model: fake_llm)
    retrieve_tool = _FakeTool("hybrid_retrieve", lambda args: f"evidence for {args['query']}")
    fake_tools = [retrieve_tool, _FakeTool("generate_answer", "combined answer")]
    monkeypatch.setattr(loop_module, "build_tools", lambda **kwargs: fake_tools)

    result = run_agent("q", session=None, vector_store=None, settings=_settings())

    assert result == "combined answer"
    assert len(retrieve_tool.invoked_with) == 2
    assert retrieve_tool.invoked_with[0]["query"] == "topic A"
    assert retrieve_tool.invoked_with[1]["query"] == "topic B"


def test_handles_tool_errors_gracefully_and_continues(monkeypatch):
    responses = [
        _FakeAIMessage(
            tool_calls=[{"name": "hybrid_retrieve", "args": {"query": "q"}, "id": "1"}]
        ),
        _FakeAIMessage(
            tool_calls=[
                {"name": "generate_answer", "args": {"question": "q", "evidence": "e"}, "id": "2"}
            ]
        ),
    ]
    fake_llm = _FakeLLM(responses)
    monkeypatch.setattr(loop_module, "get_llm", lambda model: fake_llm)

    def failing_retrieve(args):
        raise RuntimeError("boom")

    fake_tools = [
        _FakeTool("hybrid_retrieve", failing_retrieve),
        _FakeTool("generate_answer", "final answer despite error"),
    ]
    monkeypatch.setattr(loop_module, "build_tools", lambda **kwargs: fake_tools)

    result = run_agent("q", session=None, vector_store=None, settings=_settings())

    assert result == "final answer despite error"


def test_forces_final_answer_when_max_steps_exceeded(monkeypatch):
    # Never calls generate_answer on its own.
    responses = [
        _FakeAIMessage(
            tool_calls=[{"name": "hybrid_retrieve", "args": {"query": f"q{i}"}, "id": str(i)}]
        )
        for i in range(10)
    ]
    fake_llm = _FakeLLM(responses)
    monkeypatch.setattr(loop_module, "get_llm", lambda model: fake_llm)
    fake_tools = [
        _FakeTool("hybrid_retrieve", "some evidence"),
        _FakeTool("generate_answer", "forced final answer"),
    ]
    monkeypatch.setattr(loop_module, "build_tools", lambda **kwargs: fake_tools)

    result = run_agent(
        "q", session=None, vector_store=None, settings=_settings(agent_max_steps=3)
    )

    assert result == "forced final answer"
    generate_tool = fake_tools[1]
    assert generate_tool.invoked_with[0]["question"] == "q"
    assert "some evidence" in generate_tool.invoked_with[0]["evidence"]


def test_unknown_tool_call_reported_without_crashing(monkeypatch):
    responses = [
        _FakeAIMessage(tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "1"}]),
        _FakeAIMessage(
            tool_calls=[
                {"name": "generate_answer", "args": {"question": "q", "evidence": "e"}, "id": "2"}
            ]
        ),
    ]
    fake_llm = _FakeLLM(responses)
    monkeypatch.setattr(loop_module, "get_llm", lambda model: fake_llm)
    fake_tools = [_FakeTool("generate_answer", "final answer")]
    monkeypatch.setattr(loop_module, "build_tools", lambda **kwargs: fake_tools)

    result = run_agent("q", session=None, vector_store=None, settings=_settings())

    assert result == "final answer"


def test_binds_the_built_tools_and_uses_configured_model(monkeypatch):
    captured = {}
    fake_llm = _FakeLLM([_FakeAIMessage(content="done")])

    def fake_get_llm(model):
        captured["model"] = model
        return fake_llm

    monkeypatch.setattr(loop_module, "get_llm", fake_get_llm)
    fake_tools = [_FakeTool("hybrid_retrieve", "x")]
    monkeypatch.setattr(loop_module, "build_tools", lambda **kwargs: fake_tools)

    run_agent("q", session=None, vector_store=None, settings=_settings(llm_model="my-model"))

    assert captured["model"] == "my-model"
    assert fake_llm.last_bound.bound_tools == fake_tools
