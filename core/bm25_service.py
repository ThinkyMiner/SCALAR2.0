import threading
from typing import List, Dict
from rank_bm25 import BM25Plus


class BM25Service:
    """
    In-memory BM25 sparse search index.
    Rebuilt from SQLite on API startup; updated incrementally on ingest.
    Thread-safe via lock.

    Uses BM25Plus which guarantees non-negative scores, avoiding the negative
    IDF issue in BM25Okapi when a term appears in all documents.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ids: List[int] = []
        self._bm25: BM25Plus | None = None

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def add_documents(self, docs: List[Dict]):
        """
        Replace the entire index with the given docs.
        docs: list of {"id": int, "content": str}
        """
        with self._lock:
            self._ids = [d["id"] for d in docs]
            corpus = [self._tokenize(d["content"]) for d in docs]
            self._bm25 = BM25Plus(corpus) if corpus else None

    def rebuild(self, docs: List[Dict]):
        """Alias for add_documents — replaces entire index."""
        self.add_documents(docs)

    def search(self, query: str, k: int) -> List[Dict]:
        """
        Search for query, returning up to k results with score > 0.
        Returns list of {"chunk_id": int, "score": float} sorted best-first.
        """
        with self._lock:
            if self._bm25 is None or not self._ids:
                return []
            tokens = self._tokenize(query)
            scores = self._bm25.get_scores(tokens)
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]
            return [
                {"chunk_id": self._ids[i], "score": float(scores[i])}
                for i in top_indices
                if scores[i] > 0
            ]
