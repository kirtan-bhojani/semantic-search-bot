"""Document management endpoints: upload, list, and delete documents."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings
from app.dependencies import get_app_settings, get_embedding_service, get_vector_store
from app.models import DocumentInfo, DocumentListResponse
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

_SUPPORTED_CONTENT_TYPES: dict[str, str] = {
    "text/plain": "txt",
    "application/pdf": "pdf",
    "text/markdown": "md",
}


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(..., description="Text or PDF file to ingest"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_app_settings),
) -> DocumentInfo:
    """Upload a document, chunk and embed its text, and add it to the vector store.

    Supported file types: .txt, .md, .pdf
    """
    content_type = file.content_type or ""
    if content_type not in _SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {content_type}. "
                f"Supported types: {', '.join(_SUPPORTED_CONTENT_TYPES.values())}"
            ),
        )

    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb} MB",
        )

    # Extract text from file content
    text = _extract_text(content, content_type)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text content found in file")

    # Chunk → Embed → Store
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    embeddings = embedding_service.encode_batch(chunks)
    document_id = vector_store.add_document(file.filename or "untitled", chunks, embeddings)

    logger.info(
        "Uploaded '%s': %d chars → %d chunks",
        file.filename,
        len(text),
        len(chunks),
    )

    return DocumentInfo(
        document_id=document_id,
        name=file.filename or "untitled",
        chunk_count=len(chunks),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentListResponse:
    """List all ingested documents and total chunk count."""
    docs = vector_store.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                document_id=d["document_id"],
                name=d["name"],
                chunk_count=d["chunk_count"],
                ingested_at=d.get("ingested_at"),
            )
            for d in docs
        ],
        total_chunks=vector_store.total_chunks,
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    """Remove a document and all its chunks from the vector store."""
    removed = vector_store.remove_document(document_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": "Document deleted", "document_id": document_id}


def _extract_text(content: bytes, content_type: str) -> str:
    """Extract plain text from file content based on MIME type."""
    if content_type == "application/pdf":
        return _extract_pdf_text(content)
    # text/plain and text/markdown are decoded directly
    return content.decode("utf-8")


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=content, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)
