"""FAISS-based vector store with metadata persistence.

Manages a FAISS IndexIDMap for similarity search and a JSON sidecar file for
chunk metadata. Supports adding documents, searching by embedding vector,
removing documents, and persisting state to disk.

Design decisions:
- IndexFlatIP (inner product) with L2-normalized vectors gives cosine similarity.
- IndexIDMap allows assigning stable integer IDs so documents can be deleted.
- Metadata is stored in a JSON file alongside the FAISS index for simplicity.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages FAISS index and chunk metadata with disk persistence."""

    def __init__(self, store_dir: str, dimension: int = 384) -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._dimension = dimension

        self._index_path = self._store_dir / "faiss.index"
        self._metadata_path = self._store_dir / "metadata.json"

        self._index: faiss.IndexIDMap2
        self._metadata: dict

        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load an existing index from disk, or create an empty one."""
        if self._index_path.exists() and self._metadata_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                logger.info(
                    "Loaded vector store: %d vectors from %s",
                    self._index.ntotal,
                    self._store_dir,
                )
                return
            except Exception:
                logger.warning("Failed to load vector store, creating fresh index")

        self._create_empty()

    def _create_empty(self) -> None:
        """Initialize an empty FAISS index and metadata store."""
        base_index = faiss.IndexFlatIP(self._dimension)
        self._index = faiss.IndexIDMap2(base_index)
        self._metadata = {"next_id": 0, "documents": {}, "chunks": {}}
        logger.info("Created new vector store (dimension=%d)", self._dimension)

    # --- Write Operations ---

    def add_document(
        self,
        document_name: str,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> str:
        """Add a document's chunks and embeddings to the store.

        Args:
            document_name: Filename or identifier for the source document.
            chunks: List of text chunks from the document.
            embeddings: Embedding matrix of shape (len(chunks), dimension).

        Returns:
            A unique document ID string.
        """
        document_id = str(uuid.uuid4())
        num_chunks = len(chunks)
        start_id = self._metadata["next_id"]
        ids = np.arange(start_id, start_id + num_chunks, dtype=np.int64)

        # Normalize vectors so inner product equals cosine similarity
        embeddings = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(embeddings)
        self._index.add_with_ids(embeddings, ids)

        # Store document-level metadata
        self._metadata["documents"][document_id] = {
            "name": document_name,
            "chunk_count": num_chunks,
            "chunk_ids": ids.tolist(),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        # Store chunk-level metadata
        for i, chunk_id in enumerate(ids.tolist()):
            self._metadata["chunks"][str(chunk_id)] = {
                "text": chunks[i],
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
            }

        self._metadata["next_id"] = start_id + num_chunks
        self.save()

        logger.info(
            "Indexed document '%s' (%d chunks, id=%s)",
            document_name,
            num_chunks,
            document_id,
        )
        return document_id

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and all its chunks from the store.

        Args:
            document_id: The document ID returned by ``add_document``.

        Returns:
            True if the document was found and removed, False otherwise.
        """
        doc_info = self._metadata["documents"].get(document_id)
        if doc_info is None:
            return False

        chunk_ids = np.array(doc_info["chunk_ids"], dtype=np.int64)
        self._index.remove_ids(chunk_ids)

        for chunk_id in doc_info["chunk_ids"]:
            self._metadata["chunks"].pop(str(chunk_id), None)

        del self._metadata["documents"][document_id]
        self.save()

        logger.info(
            "Removed document '%s' (%d chunks)", doc_info["name"], len(chunk_ids)
        )
        return True

    # --- Read Operations ---

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: 1-D float32 array of shape (dimension,).
            k: Number of results to return.

        Returns:
            List of dicts with keys: text, document_id, document_name,
            chunk_index, score. Ordered by descending similarity.
        """
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        scores, ids = self._index.search(query, k)

        results = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id == -1:
                continue
            chunk_meta = self._metadata["chunks"].get(str(chunk_id))
            if chunk_meta is None:
                continue
            results.append(
                {
                    "text": chunk_meta["text"],
                    "document_id": chunk_meta["document_id"],
                    "document_name": chunk_meta["document_name"],
                    "chunk_index": chunk_meta["chunk_index"],
                    "score": float(score),
                }
            )

        return results

    def list_documents(self) -> list[dict]:
        """Return metadata for all ingested documents."""
        return [
            {"document_id": doc_id, **doc_info}
            for doc_id, doc_info in self._metadata["documents"].items()
        ]

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks across all documents."""
        return int(self._index.ntotal)

    # --- Persistence ---

    def save(self) -> None:
        """Write the FAISS index and metadata to disk."""
        faiss.write_index(self._index, str(self._index_path))
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)
