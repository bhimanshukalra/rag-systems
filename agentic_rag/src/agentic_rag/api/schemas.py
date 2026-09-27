from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentic_rag.persistence.models import SourceStatus


class IngestSourceRequest(BaseModel):
    source_type: Literal["web", "pdf", "markdown"]
    location: str


class SourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url_or_path: str
    source_type: str
    status: SourceStatus
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
