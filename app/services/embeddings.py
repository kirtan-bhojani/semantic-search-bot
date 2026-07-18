"""Embedding generation using sentence-transformers.

Wraps the SentenceTransformer model to provide a clean interface for encoding
single texts or batches into dense vector embeddings.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages a sentence-transformer model for generating text embeddings.

    The model is loaded once on construction and reused for all subsequent
    encoding requests, avoiding repeated model loading overhead.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dimension=%d)", self._dimension)

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a float32 embedding vector.

        Args:
            text: Input text to encode.

        Returns:
            1-D numpy array of shape (dimension,).
        """
        embedding: np.ndarray = self._model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding.astype(np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode multiple texts into embedding vectors.

        Args:
            texts: List of input texts.
            batch_size: Number of texts to encode per forward pass.

        Returns:
            2-D numpy array of shape (len(texts), dimension).
        """
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
