from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    return Settings()
