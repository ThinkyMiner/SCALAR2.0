import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_missing_api_key_returns_401(client):
    response = client.get("/stats")
    assert response.status_code == 401


def test_wrong_api_key_returns_401(client):
    response = client.get("/stats", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_valid_api_key_passes(client):
    response = client.get("/stats", headers={"X-API-Key": "test-key-abc"})
    assert response.status_code == 200


def test_health_no_auth_required(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_docs_no_auth_required(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_401_response_has_detail(client):
    response = client.get("/stats")
    assert "detail" in response.json()
