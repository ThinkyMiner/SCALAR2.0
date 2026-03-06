import os
import threading
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.database import Database


class VectorService:
    """
    Manages the FAISS vector index with stable chunk IDs from SQLite.
    Uses IndexIDMap so every vector maps to a SQLite chunk ID.

    Base index is IndexFlatIP (exact inner-product search) which supports
    remove_ids — required for soft-delete workflow.  IndexHNSWFlat does not
    implement remove_ids in the installed FAISS version.
    """

    def __init__(self, index_path: str, db: Database, model_name: str = None):
        self.index_path = index_path
        self.db = db
        self._lock = threading.Lock()

        model_name = model_name or os.getenv("SCALAR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print("Model loaded.")

        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            print(f"Loading FAISS index from {self.index_path}")
            return faiss.read_index(self.index_path)
        print("Creating new FAISS IndexIDMap(IndexFlatIP).")
        index_dir = os.path.dirname(self.index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        base = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap(base)

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    def embed(self, texts: list) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        vecs = np.array(vecs).astype("float32")
        faiss.normalize_L2(vecs)
        return vecs

    def add_vectors(self, chunk_ids: list, texts: list):
        """Embed texts and add them to the FAISS index under their SQLite IDs."""
        vecs = self.embed(texts)
        ids = np.array(chunk_ids, dtype="int64")
        with self._lock:
            self.index.add_with_ids(vecs, ids)
            self._save_index()

    def remove_vectors(self, chunk_ids: list):
        """Remove vectors by their SQLite chunk IDs."""
        ids = np.array(chunk_ids, dtype="int64")
        with self._lock:
            selector = faiss.IDSelectorArray(ids)
            self.index.remove_ids(selector)
            self._save_index()

    def search_vectors(self, query: str, k: int, candidate_multiplier: int = 5) -> list:
        """
        Return top-(k * candidate_multiplier) FAISS results for post-filtering.
        Returns list of {chunk_id, score}.
        """
        if self.index.ntotal == 0:
            return []
        query_vec = self.embed([query])
        n_candidates = min(k * candidate_multiplier, self.index.ntotal)
        distances, ids = self.index.search(query_vec, n_candidates)
        results = []
        for dist, chunk_id in zip(distances[0], ids[0]):
            if chunk_id != -1:
                results.append({"chunk_id": int(chunk_id), "score": float(dist)})
        return results
