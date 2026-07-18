"""Retrieval-Augmented Generation (RAG) question answering via Ollama.

Constructs a prompt from retrieved context chunks and sends it to a locally
running Ollama LLM instance. Uses httpx for HTTP communication instead of
a vendor SDK, keeping the integration transparent and dependency-light.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use only the information from the context to answer. If the context does not contain "
    "enough information to answer the question, say so clearly. Be concise and accurate."
)


class QAError(Exception):
    """Base exception for question-answering failures."""


class OllamaConnectionError(QAError):
    """Raised when Ollama is not reachable."""


async def generate_answer(
    query: str,
    context_chunks: list[dict],
    ollama_base_url: str,
    ollama_model: str,
) -> str:
    """Generate an answer to a question using retrieved context and an Ollama LLM.

    Args:
        query: The user's question.
        context_chunks: List of retrieved chunk dicts (must contain 'text' and
            'document_name' keys).
        ollama_base_url: Base URL of the Ollama API (e.g. http://localhost:11434).
        ollama_model: Name of the Ollama model to use (e.g. llama3.2).

    Returns:
        The generated answer string.

    Raises:
        OllamaConnectionError: If Ollama is not running or unreachable.
        QAError: If the Ollama API returns an error.
    """
    # Build context block with source attribution
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["document_name"]
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question based on the context above. "
        "Cite the source documents when relevant."
    )

    logger.info("Sending RAG query to Ollama (model=%s)", ollama_model)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_base_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    except httpx.ConnectError as exc:
        logger.error("Cannot connect to Ollama at %s", ollama_base_url)
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {ollama_base_url}. "
            "Ensure Ollama is installed and running (https://ollama.com)."
        ) from exc

    except httpx.HTTPStatusError as exc:
        logger.error("Ollama API error: %s", exc.response.text)
        raise QAError(
            f"Ollama returned an error: {exc.response.text}"
        ) from exc

    except httpx.TimeoutException as exc:
        logger.error("Ollama request timed out")
        raise QAError(
            "Ollama request timed out. The model may be loading or the query "
            "may be too complex. Try again."
        ) from exc
