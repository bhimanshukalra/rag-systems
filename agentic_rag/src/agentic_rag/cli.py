import argparse
import logging

from sqlmodel import Session

from agentic_rag.config import get_settings
from agentic_rag.generation.synthesizer import generate_answer
from agentic_rag.indexing.embeddings import get_embeddings
from agentic_rag.indexing.vector_store import get_vector_store
from agentic_rag.ingestion.pipeline import ingest_source
from agentic_rag.observability.logging import configure_logging
from agentic_rag.persistence.registry import create_engine_for

logger = logging.getLogger(__name__)


def ingest(source_type: str, location: str) -> None:
    settings = get_settings()
    engine = create_engine_for(settings.database_path)

    with Session(engine) as session:
        record = ingest_source(
            session,
            source_type=source_type,
            location=location,
            embedding_model=settings.embedding_model,
            pinecone_api_key=settings.pinecone_api_key,
            pinecone_index_name=settings.pinecone_index_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            request_timeout_seconds=settings.request_timeout_seconds,
        )

    print(
        f"[{record.status}] source {record.id}: {record.url_or_path} "
        f"({record.chunk_count} chunks)"
    )


def ask(question: str) -> None:
    settings = get_settings()
    embeddings = get_embeddings(settings.embedding_model)
    vector_store = get_vector_store(
        embeddings,
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )
    documents = vector_store.similarity_search(question, k=settings.retrieval_k)
    result = generate_answer(question, documents, model=settings.llm_model)

    print(result.answer)
    print()
    print("Sources:")
    for source in result.sources:
        print(f"  - {source}")


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(prog="agentic-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a source.")
    ingest_parser.add_argument("source_type", choices=["web", "pdf", "markdown"])
    ingest_parser.add_argument("location")

    ask_parser = subparsers.add_parser("ask", help="Ask a question.")
    ask_parser.add_argument("question")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.source_type, args.location)
    elif args.command == "ask":
        ask(args.question)


if __name__ == "__main__":
    main()
