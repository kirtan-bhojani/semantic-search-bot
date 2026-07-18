"""FastAPI dependency injection for shared services.

Provides request-scoped access to the embedding service and vector store
instances that are initialized during application startup via the lifespan.
"""

from fastapi import Request

from app.config import Settings, get_settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


def get_embedding_service(request: Request) -> EmbeddingService:
    """Retrieve the shared EmbeddingService from application state."""
    return request.app.state.embedding_service


def get_vector_store(request: Request) -> VectorStore:
    """Retrieve the shared VectorStore from application state."""
    return request.app.state.vector_store


def get_app_settings() -> Settings:
    """Retrieve application settings (cached singleton)."""
    return get_settings()
