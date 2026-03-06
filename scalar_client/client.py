"""
SCALAR Python SDK

Usage:
    from scalar_client import ScalarClient

    client = ScalarClient(base_url="http://localhost:8000", api_key="your-key")
    client.ingest("path/to/doc.pdf", namespace="research")
    results = client.search("attention mechanism", k=5, namespace="research")
    for r in results:
        print(r["source"], r["score"], r["content"][:100])
"""
import os
from typing import Dict, List, Optional

import requests


class ScalarClient:
    """HTTP client for the SCALAR Vector Database API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("SCALAR_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})

    def health(self) -> Dict:
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def stats(self) -> Dict:
        resp = self.session.get(f"{self.base_url}/stats")
        resp.raise_for_status()
        return resp.json()

    def ingest(self, file_path: str, namespace: str = "default") -> Dict:
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            resp = self.session.post(
                f"{self.base_url}/ingest/",
                params={"namespace": namespace},
                files={"file": (filename, f)},
                timeout=600,
            )
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        rerank: bool = False,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        payload: Dict = {"query_text": query, "k": k, "namespace": namespace, "rerank": rerank}
        if filter:
            payload["filter"] = filter
        resp = self.session.post(f"{self.base_url}/search/", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["results"]

    def list_documents(self, namespace: str = "default") -> List[Dict]:
        resp = self.session.get(
            f"{self.base_url}/documents/", params={"namespace": namespace}
        )
        resp.raise_for_status()
        return resp.json()["documents"]

    def delete_document(self, source: str, namespace: str = "default") -> Dict:
        resp = self.session.delete(
            f"{self.base_url}/documents/{source}", params={"namespace": namespace}
        )
        resp.raise_for_status()
        return resp.json()

    def ingest_async(self, file_path: str, namespace: str = "default") -> Dict:
        """Submit async ingest, returns {"job_id": ..., "status": "pending"}."""
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            resp = self.session.post(
                f"{self.base_url}/ingest/async",
                params={"namespace": namespace},
                files={"file": (filename, f)},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()

    def get_job(self, job_id: str) -> Dict:
        resp = self.session.get(f"{self.base_url}/ingest/jobs/{job_id}")
        resp.raise_for_status()
        return resp.json()
