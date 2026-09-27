from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_groq import ChatGroq

RETRY_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "numbered sources provided below. Cite sources inline like [1], [2] "
    "wherever you use them. If the sources don't contain the answer, say "
    "so explicitly instead of guessing."
)

NO_DOCUMENTS_ANSWER = "I don't have any relevant sources to answer that question."


@dataclass
class AnswerResult:
    answer: str
    sources: list[str]


@lru_cache(maxsize=1)
def get_llm(model: str) -> ChatGroq:
    return ChatGroq(model=model, temperature=0)


def _format_context(documents: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _unique_sources(documents: list[Document]) -> list[str]:
    seen: set[str] = set()
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def generate_answer(
    question: str, documents: list[Document], *, model: str
) -> AnswerResult:
    if not documents:
        return AnswerResult(answer=NO_DOCUMENTS_ANSWER, sources=[])

    llm = get_llm(model).with_retry(stop_after_attempt=RETRY_ATTEMPTS)
    context = _format_context(documents)

    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Sources:\n{context}\n\nQuestion: {question}"),
        ]
    )

    return AnswerResult(answer=response.content, sources=_unique_sources(documents))
