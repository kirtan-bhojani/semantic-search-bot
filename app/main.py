"""FastAPI application entry point.

Creates the application instance with lifespan management for initializing
shared services (embedding model, vector store) and registering routers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import documents, search
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    On startup: loads the embedding model and initializes (or loads) the
    FAISS vector store. Both are stored in ``app.state`` for access via
    FastAPI dependency injection.

    On shutdown: persists the vector store to disk.
    """
    settings = get_settings()

    # Initialize embedding model (loaded once, shared across requests)
    embedding_service = EmbeddingService(settings.embedding_model)
    app.state.embedding_service = embedding_service

    # Initialize vector store (loads existing index or creates empty)
    vector_store = VectorStore(
        store_dir=settings.vector_store_dir,
        dimension=embedding_service.dimension,
    )
    app.state.vector_store = vector_store

    logger.info("Semantic Search Bot ready — %d documents indexed", len(vector_store.list_documents()))
    yield

    # Persist on shutdown
    vector_store.save()
    logger.info("Vector store saved. Shutting down.")


app = FastAPI(
    title="Semantic Search Bot",
    description=(
        "A semantic document search system with embedding-based retrieval "
        "and retrieval-augmented question answering (RAG). Upload documents, "
        "search by meaning, and ask questions answered by a local LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(documents.router)
app.include_router(search.router)


@app.get("/health", tags=["System"])
async def health_check():
    """Check if the service is running."""
    return {"status": "healthy"}
