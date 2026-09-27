from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    # The agent loop doesn't return structured per-chunk sources (see the
    # trade-off documented on agent/tools.py's generate_answer wrapper),
    # so this defaults empty rather than requiring every caller to pass one.
    sources: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
