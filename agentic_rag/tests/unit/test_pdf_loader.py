import pytest

from agentic_rag.ingestion.loaders import pdf
from agentic_rag.ingestion.url_safety import UnsafeURLError


class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, _stream):
        self.pages = [_FakePage("Page one text"), _FakePage("Page two text")]


def test_load_pdf_extracts_one_document_per_page(monkeypatch):
    monkeypatch.setattr(pdf, "safe_get", lambda url, *, timeout, headers=None: _FakeResponse(b"%PDF-fake"))
    monkeypatch.setattr(pdf, "PdfReader", _FakeReader)

    docs = pdf.load_pdf("https://example.test/paper.pdf", timeout=5)

    assert len(docs) == 2
    assert docs[0].page_content == "Page one text"
    assert docs[1].page_content == "Page two text"


def test_load_pdf_sets_source_and_page_metadata(monkeypatch):
    monkeypatch.setattr(pdf, "safe_get", lambda url, *, timeout, headers=None: _FakeResponse(b"%PDF-fake"))
    monkeypatch.setattr(pdf, "PdfReader", _FakeReader)

    docs = pdf.load_pdf("https://example.test/paper.pdf", timeout=5)

    assert docs[0].metadata == {"source": "https://example.test/paper.pdf", "page": 0}
    assert docs[1].metadata == {"source": "https://example.test/paper.pdf", "page": 1}


def test_load_pdf_handles_page_with_no_extractable_text(monkeypatch):
    class _ReaderWithBlankPage:
        def __init__(self, _stream):
            self.pages = [_FakePage(None)]

    monkeypatch.setattr(pdf, "safe_get", lambda url, *, timeout, headers=None: _FakeResponse(b"%PDF-fake"))
    monkeypatch.setattr(pdf, "PdfReader", _ReaderWithBlankPage)

    docs = pdf.load_pdf("https://example.test/paper.pdf", timeout=5)

    assert docs[0].page_content == ""


def test_load_pdf_rejects_unsafe_url():
    # No mocking: validate_source_url (called inside the real safe_get)
    # must reject this before any network call is attempted.
    with pytest.raises(UnsafeURLError):
        pdf.load_pdf("http://169.254.169.254/paper.pdf", timeout=5)
