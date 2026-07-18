"""Integration tests for the FastAPI application.

These tests use FastAPI's TestClient which triggers the lifespan (loading
the embedding model). They run against a real embedding model to verify
the full pipeline works end-to-end.

Note: First run downloads the model (~80MB) and may be slow.
Subsequent runs use the cached model.
"""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with lifespan events (model loading)."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestDocumentEndpoints:
    def test_list_documents_initially_empty(self, client: TestClient):
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["documents"], list)
        assert isinstance(data["total_chunks"], int)

    def test_upload_text_document(self, client: TestClient):
        content = (
            "Machine learning is a subset of artificial intelligence. "
            "It allows systems to learn from data and improve over time."
        )
        file = io.BytesIO(content.encode("utf-8"))
        response = client.post(
            "/documents/upload",
            files={"file": ("test_doc.txt", file, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_doc.txt"
        assert data["chunk_count"] >= 1
        assert "document_id" in data

    def test_upload_rejects_unsupported_type(self, client: TestClient):
        file = io.BytesIO(b"some data")
        response = client.post(
            "/documents/upload",
            files={"file": ("data.csv", file, "text/csv")},
        )
        assert response.status_code == 400

    def test_upload_rejects_empty_file(self, client: TestClient):
        file = io.BytesIO(b"")
        response = client.post(
            "/documents/upload",
            files={"file": ("empty.txt", file, "text/plain")},
        )
        assert response.status_code == 400

    def test_delete_nonexistent_document(self, client: TestClient):
        response = client.delete("/documents/nonexistent-id")
        assert response.status_code == 404


class TestSearchEndpoints:
    def test_search_empty_store_returns_empty(self, client: TestClient):
        """Search should return empty results when no documents match."""
        response = client.post(
            "/search", json={"query": "quantum physics", "k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "quantum physics"
        assert isinstance(data["results"], list)

    def test_search_validates_empty_query(self, client: TestClient):
        response = client.post("/search", json={"query": "", "k": 5})
        assert response.status_code == 422  # Pydantic validation error

    def test_search_validates_k_range(self, client: TestClient):
        response = client.post("/search", json={"query": "test", "k": 0})
        assert response.status_code == 422

        response = client.post("/search", json={"query": "test", "k": 100})
        assert response.status_code == 422


class TestEndToEndPipeline:
    """Tests that upload a document, then search and verify results."""

    def test_upload_then_search(self, client: TestClient):
        # Upload a document about Python
        content = (
            "Python is a high-level programming language known for its "
            "readability and versatility. It is widely used in web development, "
            "data science, machine learning, and automation. Python supports "
            "multiple programming paradigms including procedural, object-oriented, "
            "and functional programming."
        )
        file = io.BytesIO(content.encode("utf-8"))
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("python_intro.txt", file, "text/plain")},
        )
        assert upload_resp.status_code == 200
        doc_id = upload_resp.json()["document_id"]

        # Search for semantically related content
        search_resp = client.post(
            "/search", json={"query": "programming language for data science", "k": 3}
        )
        assert search_resp.status_code == 200
        results = search_resp.json()["results"]
        assert len(results) >= 1

        # The result should reference our uploaded document
        assert any(r["document_name"] == "python_intro.txt" for r in results)

        # Clean up
        client.delete(f"/documents/{doc_id}")
