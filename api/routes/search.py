from typing import List, Dict
from fastapi import APIRouter, Body
from api.models import SearchQuery, SearchResponse, SearchResult
from api.main import db, vector_svc, bm25_svc

router = APIRouter()

CANDIDATE_MULTIPLIER = 10  # retrieve k*10 before filtering/reranking


def _rrf_merge(dense: List[Dict], sparse: List[Dict], k: int = 60) -> List[Dict]:
    """
    Reciprocal Rank Fusion.
    dense/sparse: list of {chunk_id, score} sorted best-first.
    Returns merged list sorted best-first by rrf_score.
    """
    scores: Dict[int, float] = {}
    for rank, item in enumerate(dense):
        scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0) + 1 / (k + rank + 1)
    for rank, item in enumerate(sparse):
        scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0) + 1 / (k + rank + 1)
    merged = [{"chunk_id": cid, "rrf_score": s} for cid, s in scores.items()]
    return sorted(merged, key=lambda x: x["rrf_score"], reverse=True)


@router.post("/", response_model=SearchResponse)
async def search(query: SearchQuery = Body(...)):
    candidates = query.k * CANDIDATE_MULTIPLIER

    # 1. Dense + sparse search
    dense_results = vector_svc.search_vectors(query.query_text, k=candidates)
    sparse_results = bm25_svc.search(query.query_text, k=candidates)

    if not dense_results and not sparse_results:
        return SearchResponse(results=[], query=query.query_text, namespace=query.namespace)

    # 2. RRF merge
    merged = _rrf_merge(dense_results, sparse_results)

    # 3. Fetch content from SQLite (deleted=0 filter applied inside)
    candidate_ids = [r["chunk_id"] for r in merged]
    chunks_by_id = {c["id"]: c for c in db.get_chunks_by_ids(candidate_ids)}

    # 4. Namespace filter + optional metadata filter
    filtered = []
    for item in merged:
        chunk = chunks_by_id.get(item["chunk_id"])
        if not chunk:
            continue
        if chunk["namespace"] != query.namespace:
            continue
        if query.filter:
            if not all(str(chunk.get(k)) == str(v) for k, v in query.filter.items()):
                continue
        filtered.append({**chunk, "rrf_score": item["rrf_score"]})

    # 5. Optional reranking (retrieve 3x k candidates for reranker to pick from)
    if query.rerank and filtered:
        from core.reranker import rerank as do_rerank
        rerank_candidates = filtered[: query.k * 3]
        top = do_rerank(query.query_text, rerank_candidates, top_k=query.k)
    else:
        top = filtered[: query.k]

    # 6. Build response
    results = [
        SearchResult(
            chunk_id=c["id"],
            source=c["source"],
            namespace=c["namespace"],
            page_number=c.get("page_number"),
            content=c["content"],
            score=c.get("rerank_score", c["rrf_score"]),
        )
        for c in top
    ]

    return SearchResponse(
        results=results,
        query=query.query_text,
        namespace=query.namespace,
    )
