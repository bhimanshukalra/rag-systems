from functools import lru_cache

from langchain_tavily import TavilySearch

RETRY_ATTEMPTS = 3


@lru_cache(maxsize=1)
def get_web_search_tool(api_key: str) -> TavilySearch:
    search_tool = TavilySearch(
        tavily_api_key=api_key,
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False,
    )
    return search_tool.with_retry(stop_after_attempt=RETRY_ATTEMPTS)


def search_web(query: str, *, api_key: str) -> str:
    """Run a web search, returning a numbered text summary of results.

    Used as the agent loop's fallback tool for questions the ingested
    corpus can't answer -- formatted as plain text since it's fed
    straight back into the agent's tool-result message.
    """
    tool = get_web_search_tool(api_key)
    response = tool.invoke({"query": query})
    results = response.get("results", [])

    if not results:
        return "No web search results found."

    parts = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "untitled")
        url = result.get("url", "unknown")
        content = result.get("content", "")
        parts.append(f"[{i}] {title} ({url})\n{content}")

    return "\n\n".join(parts)
