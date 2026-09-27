from datetime import UTC, datetime
from enum import StrEnum

from sqlmodel import Field, SQLModel


class SourceStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    FAILED = "failed"


class SourceRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    url_or_path: str
    source_type: str
    content_hash: str = Field(unique=True, index=True)
    status: SourceStatus = Field(default=SourceStatus.PENDING)
    chunk_count: int = Field(default=0)
    error_message: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
