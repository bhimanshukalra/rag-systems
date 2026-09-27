from io import BytesIO

from langchain_core.documents import Document
from pypdf import PdfReader

from agentic_rag.ingestion.url_safety import safe_get

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; agentic-rag/0.1; +local-ingestion)"


def load_pdf(url: str, *, timeout: float) -> list[Document]:
    """Fetch and parse a remote PDF into one Document per page.

    Uses safe_get() so the fetch -- and every redirect hop -- is
    validated against url_safety before any request is made.
    """
    response = safe_get(url, timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT})
    response.raise_for_status()

    reader = PdfReader(BytesIO(response.content))
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": url, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages)
    ]
