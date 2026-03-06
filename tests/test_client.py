import pytest
from unittest.mock import MagicMock, patch
from scalar_client import ScalarClient


@pytest.fixture
def client():
    return ScalarClient("http://localhost:8000", "test-key")


def test_client_sets_api_key_header(client):
    assert client.session.headers["X-API-Key"] == "test-key"


def test_client_base_url_strips_trailing_slash():
    c = ScalarClient("http://localhost:8000/", "key")
    assert c.base_url == "http://localhost:8000"


def test_health_calls_correct_endpoint(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "ok"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
        result = client.health()
        mock_get.assert_called_once_with("http://localhost:8000/health")
        assert result["status"] == "ok"


def test_search_sends_correct_payload(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        client.search("my query", k=3, namespace="ns1", rerank=True)
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["query_text"] == "my query"
        assert payload["k"] == 3
        assert payload["namespace"] == "ns1"
        assert payload["rerank"] is True


def test_search_returns_results_list(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [{"content": "hello", "score": 0.9}]}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client.session, "post", return_value=mock_resp):
        results = client.search("query")
        assert isinstance(results, list)
        assert results[0]["content"] == "hello"


def test_delete_document_calls_correct_endpoint(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "deleted"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client.session, "delete", return_value=mock_resp) as mock_del:
        client.delete_document("my_doc.pdf", namespace="test_ns")
        mock_del.assert_called_once_with(
            "http://localhost:8000/documents/my_doc.pdf",
            params={"namespace": "test_ns"},
        )


def test_list_documents_calls_correct_endpoint(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"documents": []}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
        client.list_documents(namespace="myns")
        mock_get.assert_called_once_with(
            "http://localhost:8000/documents/", params={"namespace": "myns"}
        )
