from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time
from dotenv import load_dotenv

INDEX_NAME = "industry-agentic-rag-kb"
NAMESPACE = "langgraph-agentic-rag"
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/agentic-rag"

load_dotenv()


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
    while not pinecone.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    print("Pinecone index ready:", INDEX_NAME)

    # Upload the document chunks and create the LangChain vector store.
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "namespace": NAMESPACE,
        }
    )

    print("Pinecone vector database and retriever are ready.")


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


def main():
    raw_docs = load_docs()
    chunks = get_chunks(raw_docs)
    embeddings = get_embeddings()
    setup_db(chunks, embeddings)
    retriever = get_retriever(embeddings)

    test_question = (
        "What happens if retrieved documents are not relevant in Agentic RAG?"
    )

    kb_docs = retriever.invoke(test_question)

    for i, doc in enumerate(kb_docs, 1):
        print(f"\n--- KB RESULT {i} ---")
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content[:700])


if __name__ == "__main__":
    main()
