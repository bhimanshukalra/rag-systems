import hashlib
import logging

from sqlmodel import Session

from agentic_rag.indexing.embeddings import get_embeddings
from agentic_rag.indexing.vector_store import upsert_documents
from agentic_rag.ingestion.chunking import chunk_documents
from agentic_rag.ingestion.loaders.markdown import load_markdown
from agentic_rag.ingestion.loaders.pdf import load_pdf
from agentic_rag.ingestion.loaders.web import load_web
from agentic_rag.persistence import registry
from agentic_rag.persistence.models import SourceRecord, SourceStatus

logger = logging.getLogger(__name__)

_LOADERS = {
    "web": lambda location, timeout: load_web(location, timeout=timeout),
    "pdf": lambda location, timeout: load_pdf(location, timeout=timeout),
    "markdown": lambda location, _timeout: load_markdown(location),
}


def compute_content_hash(source_type: str, location: str) -> str:
    """Identity key for registry dedup: (type, location), not fetched bytes.

    This intentionally does not detect drift in a URL's underlying content
    -- re-ingesting the same URL is a registry-level no-op once it's
    `ready`. Actual content-level idempotency happens one layer down, at
    vector_store.get_chunk_id() (hash of chunk content + position), which
    still dedups correctly even if this function's identity key is coarser.
    """
    key = f"{source_type}::{location}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ingest_source(
    session: Session,
    *,
    source_type: str,
    location: str,
    embedding_model: str,
    pinecone_api_key: str,
    pinecone_index_name: str,
    chunk_size: int,
    chunk_overlap: int,
    request_timeout_seconds: float,
) -> SourceRecord:
    if source_type not in _LOADERS:
        raise ValueError(f"unknown source_type {source_type!r}")

    content_hash = compute_content_hash(source_type, location)
    record = registry.get_or_create_source(
        session,
        url_or_path=location,
        source_type=source_type,
        content_hash=content_hash,
    )

    if record.status == SourceStatus.READY:
        logger.info("Source %s already ready, skipping re-ingestion", location)
        return record

    record = registry.mark_in_progress(session, record.id)

    try:
        raw_docs = _LOADERS[source_type](location, request_timeout_seconds)
        chunks = chunk_documents(
            raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        embeddings = get_embeddings(embedding_model)
        upsert_documents(
            chunks,
            embeddings,
            api_key=pinecone_api_key,
            index_name=pinecone_index_name,
        )
    except Exception as exc:
        # Ingestion is a job boundary: any failure here must be recorded
        # against the source, not raised and lost by a background task.
        logger.exception("Ingestion failed for %s", location)
        return registry.mark_failed(session, record.id, error=str(exc))

    return registry.mark_ready(session, record.id, chunk_count=len(chunks))
