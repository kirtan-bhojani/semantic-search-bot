"""Search and question-answering endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings
from app.dependencies import get_app_settings, get_embedding_service, get_vector_store
from app.models import AskRequest, AskResponse, SearchRequest, SearchResponse, SearchResult
from app.services.embeddings import EmbeddingService
from app.services.qa import OllamaConnectionError, QAError, generate_answer
from app.services.retrieval import retrieve
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> SearchResponse:
    """Perform semantic search over ingested documents.

    Embeds the query text and retrieves the top-k most similar document chunks
    using cosine similarity in the vector space.
    """
    results = retrieve(request.query, request.k, embedding_service, vector_store)

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
    )


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_app_settings),
) -> AskResponse:
    """Answer a question using retrieval-augmented generation (RAG).

    1. Retrieves the most relevant document chunks via semantic search.
    2. Constructs a prompt with the retrieved context.
    3. Sends the prompt to a local Ollama LLM to generate an answer.
    4. Returns the answer along with source references.

    Requires Ollama to be running locally with the configured model pulled.
    """
    # Step 1: Retrieve relevant context
    context_chunks = retrieve(
        request.question, request.k, embedding_service, vector_store
    )

    if not context_chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents have been indexed. Upload documents first.",
        )

    # Step 2-3: Generate answer via Ollama
    try:
        answer = await generate_answer(
            query=request.question,
            context_chunks=context_chunks,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
        )
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QAError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Step 4: Build response with source attribution
    sources = list({chunk["document_name"] for chunk in context_chunks})

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        context=[SearchResult(**c) for c in context_chunks],
    )
