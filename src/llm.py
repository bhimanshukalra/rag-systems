from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

RETRY_ATTEMPTS = 3


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )


def get_tavily_search_tool():
    search_tool = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False,
    )

    print("Tavily search tool ready.")

    return search_tool.with_retry(stop_after_attempt=RETRY_ATTEMPTS)


def with_retry(runnable):
    # Applied at the call site rather than inside get_llm(), since
    # with_retry()'s wrapper drops chat-model-specific methods like
    # with_structured_output().
    return runnable.with_retry(stop_after_attempt=RETRY_ATTEMPTS)
