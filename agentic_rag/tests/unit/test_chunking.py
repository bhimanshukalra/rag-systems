from langchain_core.documents import Document

from agentic_rag.ingestion.chunking import chunk_documents


def test_short_document_stays_a_single_chunk():
    doc = Document(page_content="short text", metadata={"source": "a"})

    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"


def test_long_document_splits_into_multiple_chunks():
    doc = Document(page_content="word " * 500, metadata={"source": "a"})

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_preserves_original_metadata_on_every_chunk():
    doc = Document(page_content="word " * 500, metadata={"source": "a", "page": 3})

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "a" for chunk in chunks)
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)


def test_adds_start_index_metadata():
    doc = Document(page_content="word " * 500, metadata={"source": "a"})

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    start_indices = [chunk.metadata["start_index"] for chunk in chunks]
    assert start_indices == sorted(start_indices)
    assert start_indices[0] == 0


def test_chunks_multiple_documents_independently():
    docs = [
        Document(page_content="word " * 500, metadata={"source": "a"}),
        Document(page_content="short", metadata={"source": "b"}),
    ]

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    sources = {chunk.metadata["source"] for chunk in chunks}
    assert sources == {"a", "b"}
