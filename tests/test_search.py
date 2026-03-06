import io
import pytest
from fastapi.testclient import TestClient

HEADERS = {"X-API-Key": "test-key-abc"}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def seed_search_data(client):
    """Ingest a doc once for the entire search test module."""
    content = (
        "Machine learning is a subset of artificial intelligence. "
        "It allows computers to learn from data without being explicitly programmed. "
        "Deep learning uses neural networks with many layers. "
        "Natural language processing helps computers understand human text. "
    ) * 15
    resp = client.post(
        "/ingest/?namespace=search_test",
        files={"file": ("ml_doc.txt", io.BytesIO(content.encode()), "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 200


def test_basic_search_returns_results(client):
    resp = client.post(
        "/search/",
        json={"query_text": "artificial intelligence", "k": 3, "namespace": "search_test"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) > 0
    assert data["query"] == "artificial intelligence"
    assert data["namespace"] == "search_test"


def test_search_result_has_expected_fields(client):
    resp = client.post(
        "/search/",
        json={"query_text": "neural networks", "k": 2, "namespace": "search_test"},
        headers=HEADERS,
    )
    result = resp.json()["results"][0]
    assert "chunk_id" in result
    assert "source" in result
    assert "content" in result
    assert "score" in result
    assert "namespace" in result


def test_search_respects_namespace(client):
    resp = client.post(
        "/search/",
        json={"query_text": "machine learning", "k": 5, "namespace": "nonexistent_namespace"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_k_limits_results(client):
    resp = client.post(
        "/search/",
        json={"query_text": "learning", "k": 2, "namespace": "search_test"},
        headers=HEADERS,
    )
    assert len(resp.json()["results"]) <= 2


def test_search_requires_auth(client):
    resp = client.post(
        "/search/",
        json={"query_text": "test", "k": 3, "namespace": "search_test"},
    )
    assert resp.status_code == 401


def test_search_empty_query_rejected(client):
    resp = client.post(
        "/search/",
        json={"query_text": "", "k": 3, "namespace": "search_test"},
        headers=HEADERS,
    )
    assert resp.status_code == 422
