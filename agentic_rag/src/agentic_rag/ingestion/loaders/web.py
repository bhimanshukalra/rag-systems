from bs4 import BeautifulSoup
from langchain_core.documents import Document

from agentic_rag.ingestion.url_safety import safe_get

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; agentic-rag/0.1; +local-ingestion)"


def load_web(url: str, *, timeout: float) -> list[Document]:
    """Fetch and parse a web page into a single Document.

    Uses safe_get() so the fetch -- and every redirect hop -- is
    validated against url_safety before any request is made.
    """
    response = safe_get(url, timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    return [Document(page_content=text, metadata={"source": url})]
