from langchain_core.documents import Document

from agentic_rag.generation.synthesizer import RETRY_ATTEMPTS, get_llm

# Anthropic's "Contextual Retrieval" technique: prepend a short,
# LLM-generated blurb situating each chunk within its source document,
# so an isolated chunk still carries enough context to match on -- for
# both dense embedding and BM25 keyword matching.
CONTEXT_PROMPT = (
    "You situate a chunk of text within its source document. Given a "
    "document excerpt and one chunk from it, write a single short sentence "
    "(under 30 words) stating what section/topic this chunk is from and "
    "what it covers. Do not repeat the chunk's content verbatim, and do "
    "not add commentary -- just the situating sentence."
)

# Caps the document context passed per chunk, since this is one LLM call
# per chunk and a full 80-page PDF's raw text would make each call slow
# and expensive. The opening portion of a document (title, abstract,
# intro) is usually enough for the model to situate a chunk from anywhere
# in it -- a deliberate trade-off, not an oversight.
#
# This is resent on EVERY chunk's call, not once per document, so it's
# multiplied by chunk count: at 4000 chars (~1000 tokens), an 80-chunk PDF
# alone burns ~80,000 tokens just on repeated context, which is what blew
# through Groq's 200k-tokens/day limit partway through backfilling this
# project's own corpus. 1500 chars still covers a title/abstract/intro for
# nearly any document while cutting that repeated cost roughly 2.5x.
MAX_DOCUMENT_CONTEXT_CHARS = 1500


def contextualize_chunk(document_text: str, chunk_text: str, *, model: str) -> str:
    llm = get_llm(model).with_retry(stop_after_attempt=RETRY_ATTEMPTS)
    doc_excerpt = document_text[:MAX_DOCUMENT_CONTEXT_CHARS]

    response = llm.invoke(
        [
            ("system", CONTEXT_PROMPT),
            ("human", f"Document excerpt:\n{doc_excerpt}\n\nChunk:\n{chunk_text}"),
        ]
    )

    context_blurb = response.content.strip()
    return f"{context_blurb}\n\n{chunk_text}"


def contextualize_chunks(
    chunks: list[Document], document_text: str, *, model: str
) -> list[Document]:
    """Prepend an LLM-generated context blurb to each chunk's content.

    Costs one LLM call per chunk -- opt-in via contextual_chunking_enabled,
    not run by default.
    """
    return [
        Document(
            page_content=contextualize_chunk(document_text, chunk.page_content, model=model),
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]
