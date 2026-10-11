import os

from langchain_postgres import PGVector
from langchain_core.documents import Document
from llm_helper import embeddings_model

# langchain_postgres needs the psycopg (v3) driver in the SQLAlchemy URL
DATABASE_URL = os.getenv("SUPABASE_URL") or ""


def connect_to_supabase():
    """Connect to supabase pgvector"""

    vector_store = PGVector(
        embeddings=embeddings_model,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    return vector_store


def verify_connection(vector_store: PGVector):
    """Verify the connection works"""

    test_doc = Document(page_content="This is a test document", metadata={"Test": True})

    try:
        ids = vector_store.add_documents([test_doc])
        print(f"Added test doc: {ids[0]}")

        # Search for it
        results = vector_store.similarity_search("test document")
        if results:
            print(f"Search works: {results[0].page_content}")

        vector_store.delete(ids)
        print("Cleanup complete")

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    vector_store = connect_to_supabase()
    verify_connection(vector_store)


if __name__ == "__main__":
    main()
