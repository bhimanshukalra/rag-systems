import pytest
from sqlmodel import Session

from agentic_rag.persistence import registry
from agentic_rag.persistence.models import SourceStatus
from agentic_rag.persistence.registry import create_engine_for


@pytest.fixture
def session(tmp_path):
    engine = create_engine_for(str(tmp_path / "test.db"))
    with Session(engine) as session:
        yield session


def test_get_or_create_source_creates_new_record(session):
    record = registry.get_or_create_source(
        session, url_or_path="https://example.test", source_type="web", content_hash="hash1"
    )

    assert record.id is not None
    assert record.status == SourceStatus.PENDING
    assert record.chunk_count == 0


def test_get_or_create_source_is_idempotent_by_content_hash(session):
    first = registry.get_or_create_source(
        session, url_or_path="https://example.test", source_type="web", content_hash="hash1"
    )
    second = registry.get_or_create_source(
        session,
        url_or_path="https://example.test/other-path",
        source_type="web",
        content_hash="hash1",
    )

    assert first.id == second.id


def test_mark_in_progress_updates_status(session):
    record = registry.get_or_create_source(
        session, url_or_path="a", source_type="web", content_hash="h1"
    )

    updated = registry.mark_in_progress(session, record.id)

    assert updated.status == SourceStatus.IN_PROGRESS


def test_mark_ready_sets_chunk_count_and_status(session):
    record = registry.get_or_create_source(
        session, url_or_path="a", source_type="web", content_hash="h1"
    )

    updated = registry.mark_ready(session, record.id, chunk_count=42)

    assert updated.status == SourceStatus.READY
    assert updated.chunk_count == 42


def test_mark_failed_sets_status_and_error_message(session):
    record = registry.get_or_create_source(
        session, url_or_path="a", source_type="web", content_hash="h1"
    )

    updated = registry.mark_failed(session, record.id, error="boom")

    assert updated.status == SourceStatus.FAILED
    assert updated.error_message == "boom"


def test_mark_ready_clears_previous_error_message(session):
    record = registry.get_or_create_source(
        session, url_or_path="a", source_type="web", content_hash="h1"
    )
    registry.mark_failed(session, record.id, error="boom")

    updated = registry.mark_ready(session, record.id, chunk_count=1)

    assert updated.error_message is None


def test_get_source_returns_none_for_missing_id(session):
    assert registry.get_source(session, 9999) is None


def test_list_sources_returns_all_records(session):
    registry.get_or_create_source(session, url_or_path="a", source_type="web", content_hash="h1")
    registry.get_or_create_source(session, url_or_path="b", source_type="pdf", content_hash="h2")

    sources = registry.list_sources(session)

    assert len(sources) == 2


def test_mark_in_progress_raises_for_unknown_id(session):
    with pytest.raises(ValueError):
        registry.mark_in_progress(session, 9999)
