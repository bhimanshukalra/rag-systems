import argparse

from dotenv import load_dotenv

from graph import ask_agent, build_graph
from ingestion import get_chunks, get_embeddings, get_retriever, load_docs, setup_db

load_dotenv()


def main(question: str, reingest: bool = False):
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
