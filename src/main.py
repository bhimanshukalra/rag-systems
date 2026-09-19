import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from graph import ask_agent, build_graph
from ingestion import get_chunks, get_embeddings, get_retriever, load_docs, setup_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
# Quiet chatty third-party HTTP client logging so it doesn't drown out
# our own progress messages at the default INFO level.
for _noisy_logger in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"]


def check_required_env_vars():
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]

    if missing:
        sys.exit(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in .env (see .env.example)."
        )


def main(question: str, reingest: bool = False):
    check_required_env_vars()

    embeddings = get_embeddings()

    if reingest:
        raw_docs = load_docs()
        chunks = get_chunks(raw_docs)
        retriever = setup_db(chunks, embeddings)
    else:
        retriever = get_retriever(embeddings)

    app = build_graph(retriever)

    ask_agent(app, question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        help="The question to ask the agent.",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-scrape, chunk, embed, and upload the source before querying.",
    )
    args = parser.parse_args()
    main(question=args.question, reingest=args.reingest)
