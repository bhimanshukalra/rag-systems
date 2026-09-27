from pathlib import Path

import pytest
from langchain_core.documents import Document
from sqlmodel import Session

import agentic_rag.ingestion.pipeline as pipeline_module
from agentic_rag.ingestion.pipeline import compute_content_hash, ingest_source
from agentic_rag.persistence.models import SourceStatus
from agentic_rag.persistence.registry import create_engine_for

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.md"


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def session(tmp_path):
    engine = create_engine_for(str(tmp_path / "test.db"))
    with Session(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def mock_embeddings_and_vector_store(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "get_embeddings", lambda _model_name: _FakeEmbeddings()
    )
    upserted = {"calls": []}

    def fake_upsert_documents(chunks, embeddings, *, api_key, index_name):
        upserted["calls"].append({"chunks": chunks, "index_name": index_name})
        return "vector-store"

    monkeypatch.setattr(pipeline_module, "upsert_documents", fake_upsert_documents)
    return upserted


COMMON_KWARGS = {
    "embedding_model": "fake-model",
    "pinecone_api_key": "fake-key",
    "pinecone_index_name": "fake-index",
    "chunk_size": 200,
    "chunk_overlap": 20,
    "request_timeout_seconds": 5,
}


def test_ingest_markdown_source_end_to_end(session, mock_embeddings_and_vector_store):
    record = ingest_source(
        session,
        source_type="markdown",
        location=str(FIXTURE_PATH),
        **COMMON_KWARGS,
    )

    assert record.status == SourceStatus.READY
    assert record.chunk_count > 0
    assert len(mock_embeddings_and_vector_store["calls"]) == 1


def test_ingest_dispatches_to_the_right_loader_by_source_type(monkeypatch, session):
    calls = []

    def fake_load_web(location, *, timeout):
        calls.append(("web", location, timeout))
        return [Document(page_content="web content", metadata={"source": location})]

    monkeypatch.setattr(pipeline_module, "load_web", fake_load_web)

    record = ingest_source(
        session,
        source_type="web",
        location="https://example.test/",
        **COMMON_KWARGS,
    )

    assert record.status == SourceStatus.READY
    assert calls == [("web", "https://example.test/", 5)]


def test_ingest_is_idempotent_and_skips_when_already_ready(monkeypatch, session):
    call_count = {"n": 0}

    def fake_load_markdown(location):
        call_count["n"] += 1
        return [Document(page_content="content", metadata={"source": location})]

    monkeypatch.setattr(pipeline_module, "load_markdown", fake_load_markdown)

    first = ingest_source(
        session, source_type="markdown", location=str(FIXTURE_PATH), **COMMON_KWARGS
    )
    second = ingest_source(
        session, source_type="markdown", location=str(FIXTURE_PATH), **COMMON_KWARGS
    )

    assert first.id == second.id
    assert second.status == SourceStatus.READY
    assert call_count["n"] == 1


def test_ingest_marks_failed_on_loader_error(monkeypatch, session):
    def failing_loader(location):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module, "load_markdown", failing_loader)

    record = ingest_source(
        session, source_type="markdown", location=str(FIXTURE_PATH), **COMMON_KWARGS
    )

    assert record.status == SourceStatus.FAILED
    assert "boom" in record.error_message


def test_ingest_rejects_unknown_source_type(session):
    with pytest.raises(ValueError):
        ingest_source(session, source_type="video", location="x", **COMMON_KWARGS)


def test_compute_content_hash_is_stable_and_type_sensitive():
    a = compute_content_hash("web", "https://example.test/")
    b = compute_content_hash("web", "https://example.test/")
    c = compute_content_hash("pdf", "https://example.test/")

    assert a == b
    assert a != c
