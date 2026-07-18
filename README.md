# Semantic Search Bot

A semantic document search system built with Python, FastAPI, vector embeddings, and large language models. Upload documents, search by meaning (not keywords), and ask questions answered by a locally running LLM using retrieval-augmented generation (RAG).

## Features

- **Document Ingestion** — Upload text (.txt, .md) and PDF files via REST API
- **Intelligent Chunking** — Splits documents at natural boundaries (paragraphs, sentences) with configurable overlap
- **Vector Embeddings** — Generates dense embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- **Semantic Search** — Retrieves relevant content by meaning using cosine similarity, not keyword matching
- **RAG Question Answering** — Answers questions using retrieved context + a local Ollama LLM
- **Document Management** — List and delete indexed documents
- **Persistent Storage** — FAISS index and metadata persist to disk across restarts

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI | REST API with automatic OpenAPI docs |
| Embeddings | sentence-transformers | Dense vector representations of text |
| Vector Store | FAISS | Fast similarity search over embeddings |
| LLM | Ollama (local) | Question answering via RAG — no API keys needed |
| PDF Parsing | PyMuPDF | Text extraction from PDF documents |
| HTTP Client | httpx | Async communication with Ollama |
| Configuration | pydantic-settings | Type-safe settings from environment variables |

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │               FastAPI Server                │
                    │                                             │
   Upload ─────────┤  /documents/upload                          │
                    │      │                                      │
                    │      ▼                                      │
                    │  ┌──────────┐  ┌────────────┐  ┌─────────┐ │
                    │  │ Chunking │─▶│ Embeddings │─▶│  FAISS  │ │
                    │  └──────────┘  └────────────┘  │  Index  │ │
                    │                                └────┬────┘ │
   Search ──────────┤  /search                            │      │
                    │      │         ┌────────────┐       │      │
                    │      └────────▶│ Embeddings │───────┘      │
                    │                └────────────┘   similarity  │
                    │                                  search     │
   Ask ─────────────┤  /ask                                      │
                    │      │         ┌────────────┐              │
                    │      ├────────▶│ Retrieval  │──────────┐   │
                    │      │         └────────────┘          │   │
                    │      │                                 ▼   │
                    │      │         ┌────────────┐    ┌───────┐ │
                    │      └────────▶│  Ollama    │◀───│Context│ │
                    │                │  (RAG QA)  │    └───────┘ │
                    │                └────────────┘              │
                    └─────────────────────────────────────────────┘
```

## End-to-End Pipeline

```
Upload → Extract Text → Chunk → Embed → Store in FAISS
                                              │
Query  → Embed Query  → Search FAISS ─────────┘
                              │
                              ▼
                    Retrieve Top-K Chunks
                              │
                              ▼
                    Construct Prompt with Context
                              │
                              ▼
                    Generate Answer via Ollama LLM
```

## Project Structure

```
semantic-search-bot/
├── app/
│   ├── main.py              # FastAPI app, lifespan, middleware, router registration
│   ├── config.py            # Pydantic settings (env vars, defaults)
│   ├── models.py            # Request/response Pydantic schemas
│   ├── dependencies.py      # FastAPI dependency injection
│   ├── routers/
│   │   ├── documents.py     # POST /documents/upload, GET /documents, DELETE /documents/{id}
│   │   └── search.py        # POST /search, POST /ask
│   └── services/
│       ├── chunking.py      # Text splitting with boundary detection
│       ├── embeddings.py    # SentenceTransformer wrapper
│       ├── vector_store.py  # FAISS index + metadata persistence
│       ├── retrieval.py     # Query embedding → vector search orchestration
│       └── qa.py            # RAG: context + Ollama LLM → answer
├── tests/
│   ├── test_chunking.py     # Unit tests for chunking logic
│   └── test_api.py          # Integration tests for API endpoints
├── sample_docs/             # Sample .txt files for demo
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a document (txt, md, pdf) |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{document_id}` | Remove a document from the index |
| `POST` | `/search` | Semantic search over indexed documents |
| `POST` | `/ask` | Ask a question (RAG with Ollama) |
| `GET` | `/health` | Health check |

Full interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (for question answering only — search works without it)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-search-bot.git
cd semantic-search-bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy and customize configuration
cp .env.example .env
```

### Install Ollama (for RAG)

```bash
# Install Ollama from https://ollama.com, then pull a model:
ollama pull llama3.2
```

## Running Locally

```bash
# Start the server
uvicorn app.main:app --reload

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Example Usage

### 1. Upload a Document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@sample_docs/machine_learning.txt"
```

Response:
```json
{
  "document_id": "a1b2c3d4-...",
  "name": "machine_learning.txt",
  "chunk_count": 7
}
```

### 2. Semantic Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do neural networks learn from data?", "k": 3}'
```

Response:
```json
{
  "query": "how do neural networks learn from data?",
  "results": [
    {
      "text": "Neural networks are computing systems inspired by biological...",
      "document_name": "machine_learning.txt",
      "chunk_index": 3,
      "score": 0.72
    }
  ]
}
```

### 3. Ask a Question (RAG)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between supervised and unsupervised learning?"}'
```

Response:
```json
{
  "question": "What is the difference between supervised and unsupervised learning?",
  "answer": "Supervised learning uses labeled data where the correct answer is known...",
  "sources": ["machine_learning.txt"],
  "context": [...]
}
```

### 4. List Documents

```bash
curl http://localhost:8000/documents
```

### 5. Delete a Document

```bash
curl -X DELETE http://localhost:8000/documents/{document_id}
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All settings can be configured via environment variables (prefix `SSB_`) or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `SSB_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face embedding model |
| `SSB_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `SSB_OLLAMA_MODEL` | `llama3.2` | Ollama model for RAG |
| `SSB_CHUNK_SIZE` | `512` | Max characters per chunk |
| `SSB_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `SSB_VECTOR_STORE_DIR` | `vector_store_data` | Directory for FAISS index persistence |
| `SSB_MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |

## License

MIT
