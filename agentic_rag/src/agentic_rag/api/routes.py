import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from agentic_rag.agent.loop import run_agent
from agentic_rag.api.schemas import (
    HealthResponse,
    IngestSourceRequest,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.indexing.embeddings import get_embeddings
from agentic_rag.indexing.vector_store import get_vector_store
from agentic_rag.ingestion.pipeline import ingest_source
from agentic_rag.persistence import registry
from agentic_rag.persistence.registry import create_engine_for

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=8)
def _get_engine(database_path: str):
    return create_engine_for(database_path)


def get_session(settings: Settings = Depends(get_settings)):
    engine = _get_engine(settings.database_path)
    with Session(engine) as session:
        yield session


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse()


@router.post("/sources", response_model=SourceResponse, status_code=201)
def create_source(
    payload: IngestSourceRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> SourceResponse:
    record = ingest_source(
        session,
        source_type=payload.source_type,
        location=payload.location,
        embedding_model=settings.embedding_model,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_index_name=settings.pinecone_index_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        request_timeout_seconds=settings.request_timeout_seconds,
        contextual_chunking_enabled=settings.contextual_chunking_enabled,
        llm_model=settings.llm_model,
    )
    return SourceResponse.model_validate(record)


@router.get("/sources/{source_id}", response_model=SourceResponse)
def get_source_status(
    source_id: int, session: Session = Depends(get_session)
) -> SourceResponse:
    record = registry.get_source(session, source_id)
    if record is None:
        raise HTTPException(status_code=404, detail="source not found")
    return SourceResponse.model_validate(record)


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    embeddings = get_embeddings(settings.embedding_model)
    vector_store = get_vector_store(
        embeddings,
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )
    answer = run_agent(
        payload.question, session=session, vector_store=vector_store, settings=settings
    )
    return QueryResponse(answer=answer)
