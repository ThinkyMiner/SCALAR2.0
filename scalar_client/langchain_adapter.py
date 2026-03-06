"""
LangChain VectorStore adapter for SCALAR.

Usage:
    from scalar_client.langchain_adapter import ScalarVectorStore

    store = ScalarVectorStore(
        base_url="http://localhost:8000",
        api_key="your-key",
        namespace="myns",
    )
    # Add documents (writes a temp file and ingests via SCALAR API)
    store.add_texts(["text one", "text two"])

    # Similarity search
    docs = store.similarity_search("query", k=3)

    # With scores
    docs_scores = store.similarity_search_with_score("query", k=3)

    # LangChain factory method
    store = ScalarVectorStore.from_texts(
        texts=["doc 1", "doc 2"],
        embedding=None,
        base_url="http://localhost:8000",
        api_key="your-key",
        namespace="myns",
    )
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from scalar_client.client import ScalarClient


class ScalarVectorStore(VectorStore):
    """
    LangChain-compatible VectorStore backed by the SCALAR API.
    Embedding is handled server-side; the `embedding` argument is accepted
    for LangChain compatibility but not used.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        namespace: str = "default",
        embedding: Optional[Embeddings] = None,
    ):
        self._client = ScalarClient(base_url=base_url, api_key=api_key)
        self._namespace = namespace
        self._embedding = embedding

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Write texts to a temp .txt file and ingest via SCALAR API."""
        text_list = list(texts)
        combined = "\n---\n".join(text_list)
        source_name = kwargs.get("source", "langchain_upload.txt")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(combined)
            tmp_path = f.name
        try:
            result = self._client.ingest(tmp_path, namespace=self._namespace)
        finally:
            os.remove(tmp_path)

        chunks_indexed = result.get("chunks_indexed", 0)
        return [str(i) for i in range(chunks_indexed)]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        results = self._client.search(query, k=k, namespace=self._namespace)
        return [
            Document(
                page_content=r["content"],
                metadata={
                    "source": r["source"],
                    "page_number": r.get("page_number"),
                    "score": r["score"],
                },
            )
            for r in results
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        results = self._client.search(query, k=k, namespace=self._namespace)
        return [
            (
                Document(
                    page_content=r["content"],
                    metadata={"source": r["source"], "page_number": r.get("page_number")},
                ),
                r["score"],
            )
            for r in results
        ]

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    # ------------------------------------------------------------------
    # Class factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "ScalarVectorStore":
        store = cls(
            base_url=kwargs["base_url"],
            api_key=kwargs["api_key"],
            namespace=kwargs.get("namespace", "default"),
            embedding=embedding,
        )
        store.add_texts(texts, metadatas)
        return store
