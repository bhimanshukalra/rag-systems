from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_helper import embeddings_model

document = ""


def smart_chunker(
    text: str, use_semantic: bool = True, fallback_chunk_size: int = 500
) -> list[str]:
    """
    Chunking with semantic as primary, recursive as fallback
    """

    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings=embeddings_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )
            chunks = chunker.split_text(text)

            # Validate chunks aren't too large
            max_chunk_size = 2000
            if any(len(current_chunk) > max_chunk_size for current_chunk in chunks):
                return _recursive_fallback(text, fallback_chunk_size)

            return chunks

        except Exception as e:
            print(f"Semantic chunking failed: {e}, using fallback")
            return _recursive_fallback(text, fallback_chunk_size)


def _recursive_fallback(text: str, chunk_size: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap=50)

    return splitter.split_text(text)


chunks = smart_chunker(document, use_semantic=True)
print(f"Created {len(chunks)} semantic chunks")
