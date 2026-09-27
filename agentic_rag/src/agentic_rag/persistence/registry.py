from collections.abc import Sequence
from datetime import UTC, datetime

from sqlmodel import Session, SQLModel, create_engine, select

from agentic_rag.persistence.models import SourceRecord, SourceStatus


def create_engine_for(database_path: str):
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return engine


def get_or_create_source(
    session: Session, *, url_or_path: str, source_type: str, content_hash: str
) -> SourceRecord:
    existing = session.exec(
        select(SourceRecord).where(SourceRecord.content_hash == content_hash)
    ).first()
    if existing is not None:
        return existing

    record = SourceRecord(
        url_or_path=url_or_path,
        source_type=source_type,
        content_hash=content_hash,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def mark_in_progress(session: Session, source_id: int) -> SourceRecord:
    record = _get_or_raise(session, source_id)
    record.status = SourceStatus.IN_PROGRESS
    return _save(session, record)


def mark_ready(session: Session, source_id: int, *, chunk_count: int) -> SourceRecord:
    record = _get_or_raise(session, source_id)
    record.status = SourceStatus.READY
    record.chunk_count = chunk_count
    record.error_message = None
    return _save(session, record)


def mark_failed(session: Session, source_id: int, *, error: str) -> SourceRecord:
    record = _get_or_raise(session, source_id)
    record.status = SourceStatus.FAILED
    record.error_message = error
    return _save(session, record)


def get_source(session: Session, source_id: int) -> SourceRecord | None:
    return session.get(SourceRecord, source_id)


def list_sources(session: Session) -> Sequence[SourceRecord]:
    return session.exec(select(SourceRecord)).all()


def _get_or_raise(session: Session, source_id: int) -> SourceRecord:
    record = session.get(SourceRecord, source_id)
    if record is None:
        raise ValueError(f"no source with id {source_id}")
    return record


def _save(session: Session, record: SourceRecord) -> SourceRecord:
    record.updated_at = datetime.now(UTC)
    session.add(record)
    session.commit()
    session.refresh(record)
    return record
