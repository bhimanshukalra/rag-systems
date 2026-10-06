from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agentic_rag.api.routes as routes_module
import agentic_rag.ingestion.pipeline as pipeline_module
from agentic_rag.api.app import app
from agentic_rag.api.rate_limit import _limiter_for
from agentic_rag.api.routes import _get_engine
from agentic_rag.config import Settings, get_settings

AUTH_TOKEN = "test-token"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.md"


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def settings_override(tmp_path):
    settings = Settings(
        groq_api_key="fake-groq",
        pinecone_api_key="fake-pinecone",
        api_auth_token=AUTH_TOKEN,
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
    _limiter_for.cache_clear()
    yield settings
    app.dependency_overrides.clear()
    _get_engine.cache_clear()
    _limiter_for.cache_clear()


@pytest.fixture
def client(settings_override):
    with TestClient(app, headers=AUTH_HEADERS) as test_client:
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


def test_query_returns_agent_answer(client, monkeypatch):
    captured = {}

    def fake_run_agent(question, *, session, vector_store, settings):
        captured.update(question=question, vector_store=vector_store)
        return "42"

    monkeypatch.setattr(routes_module, "get_embeddings", lambda _model: _FakeEmbeddings())
    monkeypatch.setattr(routes_module, "get_vector_store", lambda *args, **kwargs: "vector-store")
    monkeypatch.setattr(routes_module, "run_agent", fake_run_agent)

    response = client.post("/query", json={"question": "what is the answer?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "42", "sources": []}
    assert captured == {"question": "what is the answer?", "vector_store": "vector-store"}


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("post", "/sources", {"source_type": "markdown", "location": "x"}),
        ("get", "/sources/1", None),
        ("post", "/query", {"question": "q"}),
    ],
)
def test_protected_routes_reject_missing_or_wrong_token(settings_override, method, path, body):
    with TestClient(app) as unauthenticated:
        missing = getattr(unauthenticated, method)(path, **({"json": body} if body else {}))
        wrong = getattr(unauthenticated, method)(
            path,
            headers={"Authorization": "Bearer wrong"},
            **({"json": body} if body else {}),
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_healthz_does_not_require_auth(settings_override):
    with TestClient(app) as unauthenticated:
        assert unauthenticated.get("/healthz").status_code == 200


def test_rate_limit_burst_returns_429(settings_override, client):
    limit = settings_override.rate_limit_requests_per_minute
    statuses = [client.get("/sources/9999").status_code for _ in range(limit + 3)]

    assert statuses[:limit] == [404] * limit
    assert statuses[limit:] == [429] * 3
    assert int(client.get("/sources/9999").headers["Retry-After"]) >= 1


def test_unauthenticated_requests_do_not_consume_rate_limit(settings_override):
    limit = settings_override.rate_limit_requests_per_minute
    with TestClient(app) as unauthenticated:
        for _ in range(limit + 2):
            assert unauthenticated.get("/sources/1").status_code == 401
    with TestClient(app, headers=AUTH_HEADERS) as authed:
        assert authed.get("/sources/9999").status_code == 404


def test_healthz_is_not_rate_limited(settings_override, client):
    limit = settings_override.rate_limit_requests_per_minute
    assert all(client.get("/healthz").status_code == 200 for _ in range(limit + 3))


def test_query_times_out_with_504(client, settings_override, monkeypatch):
    import time

    settings_override.query_timeout_seconds = 0.2
    monkeypatch.setattr(routes_module, "get_embeddings", lambda _model: _FakeEmbeddings())
    monkeypatch.setattr(routes_module, "get_vector_store", lambda *a, **k: "vector-store")
    monkeypatch.setattr(
        routes_module, "run_agent", lambda *a, **k: time.sleep(1) or "late"
    )

    response = client.post("/query", json={"question": "slow"})

    assert response.status_code == 504
