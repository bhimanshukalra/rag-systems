from dotenv import load_dotenv

from graph import ask_agent, set_retriever
from ingestion import get_chunks, get_embeddings, get_retriever, load_docs, setup_db

load_dotenv()


def main():
    raw_docs = load_docs()
    chunks = get_chunks(raw_docs)
    embeddings = get_embeddings()
    setup_db(chunks, embeddings)
    set_retriever(get_retriever(embeddings))

    ask_agent("In Agentic RAG, what happens when retrieved documents are not relevant?")
    ask_agent(
        "What is Tavily Search and why is it useful for AI agents and RAG workflows?"
    )
    ask_agent("Hello, how are you?")
    ask_agent(
        "What is the current LangChain Tavily package used for Python web search integration?"
    )
    ask_agent("How to build a custom RAG agent with LangGraph?")


if __name__ == "__main__":
    main()
