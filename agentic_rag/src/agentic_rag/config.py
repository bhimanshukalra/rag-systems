from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings' env_file only populates this module's Settings object;
# it doesn't set real process environment variables. Some third-party
# libraries we depend on (e.g. langchain_pinecone) read PINECONE_API_KEY
# from os.environ directly, so .env must also be loaded the plain way.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-rag-kb"
    database_path: str = "./data/registry.db"
    request_timeout_seconds: int = 15
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 4
    llm_model: str = "openai/gpt-oss-20b"
    hybrid_retrieval_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
