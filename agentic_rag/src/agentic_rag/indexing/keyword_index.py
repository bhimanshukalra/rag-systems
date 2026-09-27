import json
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sqlmodel import Field, Session, SQLModel, select

from agentic_rag.indexing.vector_store import get_chunk_id

# Standard English stopwords, excluded from both indexing and querying.
# Without this, common words in a natural-language question (e.g. "what",
# "does", "for") dilute the rare, actually-discriminating terms -- "for"
# in particular isn't reliably downweighted by IDF alone in a programming
# tutorial corpus, where a real cluster of chunks is specifically about
# "for" loops.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
        "does", "doing", "don", "for", "from", "had", "has", "have", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
        "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
        "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
        "over", "own", "s", "same", "she", "should", "so", "some", "such", "t",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when",
        "where", "which", "while", "who", "whom", "why", "will", "with", "you",
        "your", "yours", "yourself", "yourselves",
    }
)


class KeywordChunk(SQLModel, table=True):
    """Persisted BM25 corpus: one row per chunk, keyed by the same chunk
    id used for Pinecone upserts, so both indexes agree on chunk identity.
    """

    id: str = Field(primary_key=True)
    source: str
    page_content: str
    metadata_json: str


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def add_chunks(session: Session, chunks: list[Document]) -> None:
    for chunk in chunks:
        chunk_id = get_chunk_id(chunk)
        record = session.get(KeywordChunk, chunk_id) or KeywordChunk(
            id=chunk_id, source="", page_content="", metadata_json="{}"
        )
        record.source = chunk.metadata.get("source", "unknown")
        record.page_content = chunk.page_content
        record.metadata_json = json.dumps(chunk.metadata)
        session.add(record)
    session.commit()


def _load_all_chunks(session: Session) -> list[Document]:
    records = session.exec(select(KeywordChunk)).all()
    return [
        Document(page_content=record.page_content, metadata=json.loads(record.metadata_json))
        for record in records
    ]


def search(session: Session, query: str, k: int) -> list[Document]:
    """BM25 search over the persisted corpus.

    Rebuilds the in-memory BM25 index from all persisted chunks on every
    call -- fine at this project's scale (low thousands of chunks, a
    single instance); worth revisiting only if the corpus grows enough
    to make that measurably slow.
    """
    documents = _load_all_chunks(session)
    if not documents:
        return []

    tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)

    # A score of 0 means zero term overlap with the query -- excluding it
    # avoids padding results with documents that share no keywords at all.
    return [documents[i] for i in ranked_indices[:k] if scores[i] > 0]
