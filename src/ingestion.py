import hashlib
import time

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "industry-agentic-rag-kb"
NAMESPACE = "langgraph-agentic-rag"
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/agentic-rag"
INDEX_READY_TIMEOUT_SECONDS = 60


def load_docs():
    loader = WebBaseLoader(
        web_paths=(SOURCE_URL,),
        requests_kwargs={
            "headers": {"User-Agent": "Mozilla/5.0 Agentic-RAG-Industry-Demo"}
        },
    )

    raw_docs = loader.load()

    print("Loaded documents:", len(raw_docs))

    return raw_docs


def get_chunks(raw_docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = splitter.split_documents(raw_docs)

    print("Total chunks:", len(chunks))

    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    sample_vector = embeddings.embed_query("What is Agentic RAG?")
    print("Embedding dimensions:", len(sample_vector))

    return embeddings


def get_chunk_id(chunk: Document) -> str:
    # Deterministic from content + position, so re-ingesting the same
    # source upserts the same vectors instead of duplicating them.
    key = f"{chunk.metadata.get('source', '')}::{chunk.metadata.get('start_index', '')}::{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def setup_db(chunks: list[Document], embeddings: HuggingFaceEmbeddings):

    # Connect to Pinecone
    pinecone = Pinecone()

    # Create the index only if it does not already exist.
    existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    # Wait until Pinecone reports the new index as ready.
    deadline = time.monotonic() + INDEX_READY_TIMEOUT_SECONDS
    while not pinecone.describe_index(INDEX_NAME).status["ready"]:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Pinecone index {INDEX_NAME!r} did not become ready within "
                f"{INDEX_READY_TIMEOUT_SECONDS} seconds."
            )
        time.sleep(1)

    print("Pinecone index ready:", INDEX_NAME)

    # Upload the document chunks and create the LangChain vector store.
    # Stable ids make this an upsert: re-running ingestion overwrites
    # existing vectors instead of adding duplicates.
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        ids=[get_chunk_id(chunk) for chunk in chunks],
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "namespace": NAMESPACE,
        }
    )

    print("Pinecone vector database and retriever are ready.")

    return retriever


def get_retriever(embeddings: HuggingFaceEmbeddings):
    # Connect to Pinecone
    pinecone = Pinecone()

    # Load existing Pinecone index
    index = pinecone.Index(INDEX_NAME)

    # Connect existing index with LangChain
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "namespace": NAMESPACE,
        }
    )

    return retriever
