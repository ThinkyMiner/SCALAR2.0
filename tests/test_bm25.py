import pytest
from core.bm25_service import BM25Service


@pytest.fixture
def bm25():
    svc = BM25Service()
    svc.add_documents([
        {"id": 1, "content": "the cat sat on the mat"},
        {"id": 2, "content": "the dog ran in the park"},
        {"id": 3, "content": "cats and dogs are pets"},
    ])
    return svc


def test_search_returns_relevant_results(bm25):
    results = bm25.search("cat", k=2)
    ids = [r["chunk_id"] for r in results]
    assert 1 in ids  # "cat" is in doc 1


def test_search_returns_scores(bm25):
    results = bm25.search("dog", k=3)
    assert all("score" in r for r in results)
    assert all(r["score"] >= 0 for r in results)


def test_rebuild_clears_previous(bm25):
    bm25.rebuild([{"id": 10, "content": "something new"}])
    results = bm25.search("new", k=1)
    assert results[0]["chunk_id"] == 10


def test_empty_index_returns_empty():
    empty = BM25Service()
    results = empty.search("anything", k=5)
    assert results == []


def test_search_k_limits_results(bm25):
    results = bm25.search("the", k=2)
    assert len(results) <= 2


def test_results_sorted_by_score_descending(bm25):
    results = bm25.search("cat mat", k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_zero_score_results_excluded(bm25):
    # Query that matches nothing should return empty or only nonzero scores
    results = bm25.search("xyzzy_no_match_token", k=3)
    assert all(r["score"] > 0 for r in results)
