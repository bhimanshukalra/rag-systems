import pytest
from langchain_core.documents import Document
from sqlmodel import Session

from agentic_rag.indexing.keyword_index import KeywordChunk, add_chunks, search
from agentic_rag.indexing.vector_store import get_chunk_id
from agentic_rag.persistence.registry import create_engine_for


@pytest.fixture
def session(tmp_path):
    engine = create_engine_for(str(tmp_path / "test.db"))
    with Session(engine) as session:
        yield session


def test_add_chunks_persists_page_content_and_source(session):
    doc = Document(page_content="widgets are great", metadata={"source": "a.md"})

    add_chunks(session, [doc])

    record = session.get(KeywordChunk, get_chunk_id(doc))
    assert record.page_content == "widgets are great"
    assert record.source == "a.md"


def test_add_chunks_upserts_same_chunk_id_instead_of_duplicating(session):
    doc = Document(page_content="original text", metadata={"source": "a.md", "start_index": 0})
    add_chunks(session, [doc])

    updated = Document(page_content="updated text", metadata={"source": "a.md", "start_index": 0})
    add_chunks(session, [updated])

    record = session.get(KeywordChunk, get_chunk_id(updated))
    assert record.page_content == "updated text"


def test_search_ranks_by_term_overlap(session):
    docs = [
        Document(page_content="the widget deluxe model has a warranty", metadata={"source": "products.md"}),
        Document(page_content="python interpreter starts in interactive mode", metadata={"source": "intro.html"}),
        Document(page_content="self attention relates positions in a sequence", metadata={"source": "attention.pdf"}),
    ]
    add_chunks(session, docs)

    results = search(session, "python interpreter interactive", k=3)

    assert len(results) >= 1
    assert results[0].metadata["source"] == "intro.html"


def test_search_excludes_zero_overlap_documents(session):
    docs = [
        Document(page_content="the widget deluxe model has a warranty", metadata={"source": "products.md"}),
    ]
    add_chunks(session, docs)

    results = search(session, "completely unrelated query xyz", k=5)

    assert results == []


def test_search_returns_empty_list_when_index_is_empty(session):
    results = search(session, "anything", k=5)

    assert results == []


def test_search_respects_k(session):
    docs = [
        Document(page_content=f"widget widget widget number {i}", metadata={"source": f"s{i}.md"})
        for i in range(5)
    ]
    add_chunks(session, docs)

    results = search(session, "widget", k=2)

    assert len(results) == 2
