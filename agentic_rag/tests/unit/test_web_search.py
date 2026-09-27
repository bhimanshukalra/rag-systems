import agentic_rag.retrieval.web_search as web_search_module
from agentic_rag.retrieval.web_search import get_web_search_tool, search_web


class _FakeRetryable:
    def __init__(self, response):
        self._response = response
        self.invoked_with = None
        self.stop_after_attempt = None

    def invoke(self, payload):
        self.invoked_with = payload
        return self._response


class _FakeTavilySearch:
    def __init__(self, *, tavily_api_key, max_results, topic, include_answer, include_raw_content):
        self.tavily_api_key = tavily_api_key
        self.retryable = _FakeRetryable({"results": []})

    def with_retry(self, *, stop_after_attempt):
        self.retryable.stop_after_attempt = stop_after_attempt
        return self.retryable


def _patch_tavily(monkeypatch):
    monkeypatch.setattr(web_search_module, "TavilySearch", _FakeTavilySearch)
    get_web_search_tool.cache_clear()


def test_search_web_formats_numbered_results(monkeypatch):
    _patch_tavily(monkeypatch)
    tool = get_web_search_tool("fake-key")
    tool._response = {
        "results": [
            {"title": "Python 3.13 Release", "url": "https://example.test/a", "content": "Released in October."},
            {"title": "Another Result", "url": "https://example.test/b", "content": "More details here."},
        ]
    }

    result = search_web("latest python version", api_key="fake-key")

    assert "[1] Python 3.13 Release (https://example.test/a)" in result
    assert "Released in October." in result
    assert "[2] Another Result (https://example.test/b)" in result

    get_web_search_tool.cache_clear()


def test_search_web_passes_query_through(monkeypatch):
    _patch_tavily(monkeypatch)
    tool = get_web_search_tool("fake-key")

    search_web("what is the weather", api_key="fake-key")

    assert tool.invoked_with == {"query": "what is the weather"}

    get_web_search_tool.cache_clear()


def test_search_web_handles_no_results(monkeypatch):
    _patch_tavily(monkeypatch)

    result = search_web("nothing found query", api_key="fake-key")

    assert result == "No web search results found."

    get_web_search_tool.cache_clear()


def test_search_web_uses_retry(monkeypatch):
    _patch_tavily(monkeypatch)

    search_web("q", api_key="fake-key")

    tool = get_web_search_tool("fake-key")
    assert tool.stop_after_attempt == 3

    get_web_search_tool.cache_clear()


def test_get_web_search_tool_is_cached_per_api_key(monkeypatch):
    calls = {"n": 0}

    class _CountingFake(_FakeTavilySearch):
        def __init__(self, **kwargs):
            calls["n"] += 1
            super().__init__(**kwargs)

    monkeypatch.setattr(web_search_module, "TavilySearch", _CountingFake)
    get_web_search_tool.cache_clear()

    get_web_search_tool("fake-key")
    get_web_search_tool("fake-key")

    assert calls["n"] == 1

    get_web_search_tool.cache_clear()
