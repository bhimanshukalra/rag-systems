from pathlib import Path

from langchain_core.documents import Document


def load_markdown(path: str | Path) -> list[Document]:
    """Load a local markdown file as a single Document.

    This is the one Phase 0 loader with no network dependency: the path
    is a local file, so `url_safety` checks don't apply here.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(file_path)})]
