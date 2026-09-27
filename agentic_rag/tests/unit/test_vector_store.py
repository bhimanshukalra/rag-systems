from typing import ClassVar

import pytest
from langchain_core.documents import Document

import agentic_rag.indexing.vector_store as vector_store_module
from agentic_rag.indexing.vector_store import (
    get_chunk_id,
    get_vector_store,
    upsert_documents,
)


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _FakeStatus:
    def __init__(self, ready):
        self.status = {"ready": ready}


# -- get_chunk_id -------------------------------------------------------------


def test_chunk_id_is_deterministic():
    doc = Document(page_content="hello", metadata={"source": "a", "start_index": 0})
    assert get_chunk_id(doc) == get_chunk_id(doc)


def test_chunk_id_differs_by_content():
    doc1 = Document(page_content="hello", metadata={"source": "a", "start_index": 0})
    doc2 = Document(page_content="world", metadata={"source": "a", "start_index": 0})
    assert get_chunk_id(doc1) != get_chunk_id(doc2)


def test_chunk_id_differs_by_position():
    doc1 = Document(page_content="hello", metadata={"source": "a", "start_index": 0})
    doc2 = Document(page_content="hello", metadata={"source": "a", "start_index": 10})
    assert get_chunk_id(doc1) != get_chunk_id(doc2)


# -- upsert_documents ----------------------------------------------------------


def test_upsert_creates_index_when_missing(monkeypatch):
    created = []

    class FakePinecone:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def list_indexes(self):
            return []

        def create_index(self, *, name, dimension, metric, spec):
            created.append({"name": name, "dimension": dimension})

        def describe_index(self, name):
            return _FakeStatus(ready=True)

    class FakeVectorStore:
        calls: ClassVar[list] = []

        @classmethod
        def from_documents(cls, *, documents, embedding, index_name, ids, pinecone_api_key):
            cls.calls.append(
                {"index_name": index_name, "ids": ids, "pinecone_api_key": pinecone_api_key}
            )
            return "vector-store"

    monkeypatch.setattr(vector_store_module, "Pinecone", FakePinecone)
    monkeypatch.setattr(vector_store_module, "PineconeVectorStore", FakeVectorStore)

    docs = [Document(page_content="hello", metadata={"source": "a", "start_index": 0})]

    result = upsert_documents(docs, _FakeEmbeddings(), api_key="key", index_name="my-index")

    assert result == "vector-store"
    assert created == [{"name": "my-index", "dimension": 3}]
    assert FakeVectorStore.calls[-1]["index_name"] == "my-index"
    assert FakeVectorStore.calls[-1]["ids"] == [get_chunk_id(docs[0])]
    # PineconeVectorStore.from_documents() builds its own internal Pinecone
    # client -- it does NOT reuse the one constructed above for index
    # creation, so the api_key must be passed here explicitly too.
    assert FakeVectorStore.calls[-1]["pinecone_api_key"] == "key"


def test_upsert_skips_create_when_index_already_exists(monkeypatch):
    created = []

    class FakePinecone:
        def __init__(self, api_key=None):
            pass

        def list_indexes(self):
            return [{"name": "my-index"}]

        def create_index(self, **kwargs):
            created.append(kwargs)

        def describe_index(self, name):
            return _FakeStatus(ready=True)

    class FakeVectorStore:
        @classmethod
        def from_documents(cls, **kwargs):
            return "vector-store"

    monkeypatch.setattr(vector_store_module, "Pinecone", FakePinecone)
    monkeypatch.setattr(vector_store_module, "PineconeVectorStore", FakeVectorStore)

    docs = [Document(page_content="hello", metadata={"source": "a", "start_index": 0})]
    upsert_documents(docs, _FakeEmbeddings(), api_key="key", index_name="my-index")

    assert created == []


def test_upsert_raises_if_index_never_becomes_ready(monkeypatch):
    class FakePinecone:
        def __init__(self, api_key=None):
            pass

        def list_indexes(self):
            return []

        def create_index(self, **kwargs):
            pass

        def describe_index(self, name):
            return _FakeStatus(ready=False)

    monkeypatch.setattr(vector_store_module, "Pinecone", FakePinecone)

    # Fast-forward past the timeout instead of actually waiting on it.
    fake_clock = iter([0, 61])
    monkeypatch.setattr(
        vector_store_module.time, "monotonic", lambda: next(fake_clock, 61)
    )
    monkeypatch.setattr(vector_store_module.time, "sleep", lambda _seconds: None)

    docs = [Document(page_content="hello", metadata={"source": "a", "start_index": 0})]

    with pytest.raises(TimeoutError):
        upsert_documents(docs, _FakeEmbeddings(), api_key="key", index_name="my-index")


# -- get_vector_store ----------------------------------------------------------


def test_get_vector_store_connects_to_existing_index(monkeypatch):
    class FakePinecone:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def Index(self, name):
            return f"index:{name}"

    class FakeVectorStore:
        def __init__(self, *, index, embedding):
            self.index = index
            self.embedding = embedding

    monkeypatch.setattr(vector_store_module, "Pinecone", FakePinecone)
    monkeypatch.setattr(vector_store_module, "PineconeVectorStore", FakeVectorStore)

    store = get_vector_store(_FakeEmbeddings(), api_key="key", index_name="my-index")

    assert isinstance(store, FakeVectorStore)
    assert store.index == "index:my-index"
