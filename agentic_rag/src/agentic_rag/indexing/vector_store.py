import hashlib
import logging
import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

INDEX_READY_TIMEOUT_SECONDS = 60


def get_chunk_id(chunk: Document) -> str:
    # Deterministic from content + position, so re-ingesting the same
    # source upserts the same vectors instead of duplicating them.
    key = (
        f"{chunk.metadata.get('source', '')}::"
        f"{chunk.metadata.get('start_index', '')}::"
        f"{chunk.page_content}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _ensure_index_ready(pinecone: Pinecone, index_name: str, dimension: int) -> None:
    existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]

    if index_name not in existing_indexes:
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    deadline = time.monotonic() + INDEX_READY_TIMEOUT_SECONDS
    while not pinecone.describe_index(index_name).status["ready"]:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Pinecone index {index_name!r} did not become ready within "
                f"{INDEX_READY_TIMEOUT_SECONDS} seconds."
            )
        time.sleep(1)


def upsert_documents(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    *,
    api_key: str,
    index_name: str,
) -> PineconeVectorStore:
    pinecone = Pinecone(api_key=api_key)

    dimension = len(embeddings.embed_query("dimension probe"))
    _ensure_index_ready(pinecone, index_name, dimension)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        ids=[get_chunk_id(chunk) for chunk in chunks],
    )

    logger.info("Upserted %d chunks into Pinecone index %s", len(chunks), index_name)

    return vector_store


def get_vector_store(
    embeddings: HuggingFaceEmbeddings, *, api_key: str, index_name: str
) -> PineconeVectorStore:
    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(index_name)

    return PineconeVectorStore(index=index, embedding=embeddings)
