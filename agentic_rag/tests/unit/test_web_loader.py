import pytest

from agentic_rag.ingestion.loaders import web
from agentic_rag.ingestion.url_safety import UnsafeURLError


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_load_web_parses_html_and_strips_script_style(monkeypatch):
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><h1>Hello</h1><script>alert('x')</script><p>World</p></body></html>"
    )

    def fake_safe_get(url, *, timeout, headers=None):
        return _FakeResponse(html)

    monkeypatch.setattr(web, "safe_get", fake_safe_get)

    docs = web.load_web("https://example.test/", timeout=5)

    assert len(docs) == 1
    assert "Hello" in docs[0].page_content
    assert "World" in docs[0].page_content
    assert "alert" not in docs[0].page_content
    assert "color:red" not in docs[0].page_content


def test_load_web_sets_source_metadata(monkeypatch):
    def fake_safe_get(url, *, timeout, headers=None):
        return _FakeResponse("<html><body>content</body></html>")

    monkeypatch.setattr(web, "safe_get", fake_safe_get)

    docs = web.load_web("https://example.test/page", timeout=5)

    assert docs[0].metadata["source"] == "https://example.test/page"


def test_load_web_rejects_unsafe_url():
    # No mocking: validate_source_url (called inside the real safe_get)
    # must reject this before any network call is attempted.
    with pytest.raises(UnsafeURLError):
        web.load_web("http://169.254.169.254/", timeout=5)
