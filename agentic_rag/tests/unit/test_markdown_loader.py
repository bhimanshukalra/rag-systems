from pathlib import Path

import pytest

from agentic_rag.ingestion.loaders.markdown import load_markdown

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.md"


def test_loads_fixture_content():
    docs = load_markdown(FIXTURE_PATH)

    assert len(docs) == 1
    assert "Vacation Policy" in docs[0].page_content
    assert "23 paid vacation days" in docs[0].page_content


def test_sets_source_metadata():
    docs = load_markdown(FIXTURE_PATH)

    assert docs[0].metadata["source"] == str(FIXTURE_PATH)


def test_accepts_string_path():
    docs = load_markdown(str(FIXTURE_PATH))

    assert len(docs) == 1


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_markdown(FIXTURE_PATH.parent / "does-not-exist.md")
