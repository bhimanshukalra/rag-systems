import argparse

from dotenv import load_dotenv

from graph import ask_agent, build_graph
from ingestion import get_chunks, get_embeddings, get_retriever, load_docs, setup_db

load_dotenv()


def main(reingest: bool = False):
    embeddings = get_embeddings()

    if reingest:
        raw_docs = load_docs()
        chunks = get_chunks(raw_docs)
        retriever = setup_db(chunks, embeddings)
    else:
        retriever = get_retriever(embeddings)

    app = build_graph(retriever)

    demo_questions = [
        "In Agentic RAG, what happens when retrieved documents are not relevant?",
        "What is Tavily Search and why is it useful for AI agents and RAG workflows?",
        "Hello, how are you?",
        "What is the current LangChain Tavily package used for Python web search integration?",
        "How to build a custom RAG agent with LangGraph?",
    ]

    for question in demo_questions:
        ask_agent(app, question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-scrape, chunk, embed, and upload the source before querying.",
    )
    args = parser.parse_args()
    main(reingest=args.reingest)
