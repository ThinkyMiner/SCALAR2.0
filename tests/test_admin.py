import pytest
from fastapi.testclient import TestClient

HEADERS = {"X-API-Key": "test-key-abc"}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_stats_returns_expected_fields(client):
    resp = client.get("/stats", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_chunks" in data
    assert "total_sources" in data
    assert "index_vector_count" in data
    assert "embedding_model" in data


def test_backup_returns_zip(client):
    resp = client.get("/admin/backup", headers=HEADERS)
    assert resp.status_code == 200
    assert "zip" in resp.headers["content-type"]


def test_stats_requires_auth(client):
    resp = client.get("/stats")
    assert resp.status_code == 401
