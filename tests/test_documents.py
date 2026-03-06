import io
import pytest
from fastapi.testclient import TestClient

HEADERS = {"X-API-Key": "test-key-abc"}
TXT = b"Document management test content for indexing purposes. " * 15


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def ingest(client, name, ns="docns"):
    return client.post(
        f"/ingest/?namespace={ns}",
        files={"file": (name, io.BytesIO(TXT), "text/plain")},
        headers=HEADERS,
    )


def test_list_documents_returns_ingested(client):
    ingest(client, "list_me.txt", "listns")
    resp = client.get("/documents/?namespace=listns", headers=HEADERS)
    assert resp.status_code == 200
    sources = [d["source"] for d in resp.json()["documents"]]
    assert "list_me.txt" in sources


def test_delete_document_removes_it(client):
    ingest(client, "del_me.txt", "delns")
    resp = client.delete("/documents/del_me.txt?namespace=delns", headers=HEADERS)
    assert resp.status_code == 200
    resp2 = client.get("/documents/?namespace=delns", headers=HEADERS)
    sources = [d["source"] for d in resp2.json()["documents"]]
    assert "del_me.txt" not in sources


def test_delete_nonexistent_returns_404(client):
    resp = client.delete("/documents/ghost.txt?namespace=default", headers=HEADERS)
    assert resp.status_code == 404


def test_update_document_replaces_content(client):
    ingest(client, "update_me.txt", "updns")
    new_content = b"Brand new content after update operation. " * 15
    resp = client.put(
        "/documents/update_me.txt?namespace=updns",
        files={"file": ("update_me.txt", io.BytesIO(new_content), "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"


def test_list_requires_auth(client):
    resp = client.get("/documents/")
    assert resp.status_code == 401
