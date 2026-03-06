import os
from typing import List, Dict

_MODEL_NAME = os.getenv("SCALAR_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker: {_MODEL_NAME}")
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def rerank(query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
    """
    Rerank chunks using cross-encoder. Returns top_k results with 'rerank_score' added.
    chunks must have a 'content' key.
    """
    if not chunks:
        return []
    model = get_reranker()
    pairs = [(query, c["content"]) for c in chunks]
    scores = model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]
