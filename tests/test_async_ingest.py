import io
import pytest
from fastapi.testclient import TestClient

HEADERS = {"X-API-Key": "test-key-abc"}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_async_ingest_returns_job_id(client):
    resp = client.post(
        "/ingest/async?namespace=asyncns",
        files={"file": ("async1.txt", io.BytesIO(b"async content " * 20), "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["source"] == "async1.txt"


def test_job_status_accessible(client):
    resp = client.post(
        "/ingest/async?namespace=jobns",
        files={"file": ("async2.txt", io.BytesIO(b"job status content " * 20), "text/plain")},
        headers=HEADERS,
    )
    job_id = resp.json()["job_id"]
    status_resp = client.get(f"/ingest/jobs/{job_id}", headers=HEADERS)
    assert status_resp.status_code == 200
    assert status_resp.json()["id"] == job_id


def test_unknown_job_returns_404(client):
    resp = client.get("/ingest/jobs/no-such-id-xyz", headers=HEADERS)
    assert resp.status_code == 404


def test_async_ingest_requires_auth(client):
    resp = client.post(
        "/ingest/async",
        files={"file": ("noauth.txt", io.BytesIO(b"data"), "text/plain")},
    )
    assert resp.status_code == 401
