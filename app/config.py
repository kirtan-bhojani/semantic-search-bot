"""Centralized application configuration via environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables prefixed with SSB_.

    All settings have sensible defaults and can be overridden via a .env file
    or environment variables. For example, set SSB_OLLAMA_MODEL=mistral to
    switch the LLM used for question answering.
    """

    # Embedding model (Hugging Face model identifier)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store persistence directory
    vector_store_dir: str = "vector_store_data"

    # Text chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Ollama LLM configuration (for RAG question answering)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # CORS
    cors_origins: list[str] = ["*"]

    # Upload limits
    max_file_size_mb: int = 50

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SSB_")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance. Cached so the .env file is read once."""
    return Settings()
