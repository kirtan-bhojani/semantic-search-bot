"""Semantic retrieval pipeline.

Orchestrates the search flow: query text → embedding → vector similarity search.
This module is the glue between the embedding service and the vector store,
keeping each of those services focused on a single responsibility.
"""

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


def retrieve(
    query: str,
    k: int,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
) -> list[dict]:
    """Retrieve the top-k semantically similar chunks for a natural language query.

    Args:
        query: Natural language search query.
        k: Number of results to return.
        embedding_service: Service for generating query embeddings.
        vector_store: Vector store to search against.

    Returns:
        List of result dicts ordered by descending similarity score.
        Each dict contains: text, document_id, document_name, chunk_index, score.
    """
    query_embedding = embedding_service.encode(query)
    return vector_store.search(query_embedding, k)
