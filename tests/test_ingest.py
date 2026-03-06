import io
import pytest
from fastapi.testclient import TestClient

HEADERS = {"X-API-Key": "test-key-abc"}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def make_txt(content: str = "This is test content for semantic search. " * 10) -> io.BytesIO:
    return io.BytesIO(content.encode())


def test_ingest_txt_succeeds(client):
    resp = client.post(
        "/ingest/?namespace=ingest_test",
        files={"file": ("ingest_basic.txt", make_txt(), "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] > 0
    assert data["source"] == "ingest_basic.txt"
    assert data["namespace"] == "ingest_test"


def test_ingest_duplicate_rejected(client):
    content = make_txt("duplicate content test " * 10)
    client.post(
        "/ingest/?namespace=dup_ns",
        files={"file": ("dup_file.txt", content, "text/plain")},
        headers=HEADERS,
    )
    content.seek(0)
    resp = client.post(
        "/ingest/?namespace=dup_ns",
        files={"file": ("dup_file.txt", content, "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 422


def test_ingest_unsupported_type_rejected(client):
    resp = client.post(
        "/ingest/",
        files={"file": ("bad.xyz", io.BytesIO(b"data"), "application/octet-stream")},
        headers=HEADERS,
    )
    assert resp.status_code == 400


def test_ingest_custom_namespace(client):
    resp = client.post(
        "/ingest/?namespace=custom_ns",
        files={"file": ("ns_test.txt", make_txt("namespace test content " * 10), "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["namespace"] == "custom_ns"


def test_ingest_requires_auth(client):
    resp = client.post(
        "/ingest/",
        files={"file": ("auth_test.txt", make_txt(), "text/plain")},
    )
    assert resp.status_code == 401


def test_ingest_md_file(client):
    md_content = "# Header\n\nSome markdown text here. " * 10
    resp = client.post(
        "/ingest/?namespace=md_ns",
        files={"file": ("test.md", io.BytesIO(md_content.encode()), "text/markdown")},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] > 0
