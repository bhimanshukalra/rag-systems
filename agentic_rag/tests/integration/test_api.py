from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import agentic_rag.api.routes as routes_module
import agentic_rag.ingestion.pipeline as pipeline_module
from agentic_rag.api.app import app
from agentic_rag.api.routes import _get_engine
from agentic_rag.config import Settings, get_settings

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.md"


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def settings_override(tmp_path):
    settings = Settings(
        groq_api_key="fake-groq",
        pinecone_api_key="fake-pinecone",
        pinecone_index_name="fake-index",
        database_path=str(tmp_path / "test.db"),
        request_timeout_seconds=5,
        embedding_model="fake-model",
        chunk_size=200,
        chunk_overlap=20,
        retrieval_k=4,
        llm_model="fake-llm",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    _get_engine.cache_clear()
    yield settings
    app.dependency_overrides.clear()
    _get_engine.cache_clear()


@pytest.fixture
def client(settings_override):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def mock_pipeline_embeddings_and_vector_store(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_embeddings", lambda _model: _FakeEmbeddings())
    monkeypatch.setattr(
        pipeline_module,
        "upsert_documents",
        lambda chunks, embeddings, *, api_key, index_name: "vector-store",
    )


def test_healthz(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_source_ingests_markdown_fixture(client):
    response = client.post(
        "/sources", json={"source_type": "markdown", "location": str(FIXTURE_PATH)}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] > 0


def test_create_source_rejects_invalid_source_type(client):
    response = client.post("/sources", json={"source_type": "video", "location": "x"})

    assert response.status_code == 422


def test_get_source_status_after_ingest(client):
    create_response = client.post(
        "/sources", json={"source_type": "markdown", "location": str(FIXTURE_PATH)}
    )
    source_id = create_response.json()["id"]

    response = client.get(f"/sources/{source_id}")

    assert response.status_code == 200
    assert response.json()["id"] == source_id
    assert response.json()["status"] == "ready"


def test_get_source_status_returns_404_for_unknown_id(client):
    response = client.get("/sources/9999")

    assert response.status_code == 404


def test_query_returns_answer_and_sources(client, monkeypatch):
    class _FakeVectorStore:
        def similarity_search(self, question, k):
            return [
                Document(page_content="the answer is 42", metadata={"source": "s1"})
            ]

    class _FakeResult:
        answer = "42"
        sources: ClassVar[list[str]] = ["s1"]

    monkeypatch.setattr(routes_module, "get_embeddings", lambda _model: _FakeEmbeddings())
    monkeypatch.setattr(
        routes_module, "get_vector_store", lambda *args, **kwargs: _FakeVectorStore()
    )
    monkeypatch.setattr(
        routes_module,
        "generate_answer",
        lambda question, docs, *, model: _FakeResult(),
    )

    response = client.post("/query", json={"question": "what is the answer?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "42", "sources": ["s1"]}
