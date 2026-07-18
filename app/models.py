"""Pydantic schemas for API request and response bodies."""

from pydantic import BaseModel, Field


# --- Search ---


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class SearchResult(BaseModel):
    text: str = Field(description="Chunk text content")
    document_id: str = Field(description="Source document identifier")
    document_name: str = Field(description="Source document filename")
    chunk_index: int = Field(description="Position of chunk within the document")
    score: float = Field(description="Cosine similarity score (higher is more relevant)")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# --- Question Answering ---


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    k: int = Field(
        default=5, ge=1, le=20, description="Number of context chunks to retrieve"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = Field(description="Source documents used to generate the answer")
    context: list[SearchResult] = Field(description="Retrieved context chunks")


# --- Documents ---


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    chunk_count: int
    ingested_at: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int
