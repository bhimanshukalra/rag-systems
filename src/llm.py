from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


def get_llm():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    return llm


def get_tavily_search_tool():
    search_tool = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False,
    )

    print("Tavily search tool ready.")

    return search_tool
