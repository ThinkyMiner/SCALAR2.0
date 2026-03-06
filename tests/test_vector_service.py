import pytest
import numpy as np
from core.database import Database
from core.vector_service import VectorService


@pytest.fixture
def svc(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    return VectorService(
        index_path=str(tmp_path / "index.faiss"),
        db=db
    )


def test_embed_returns_normalized_vectors(svc):
    vecs = svc.embed(["hello world", "test text"])
    assert vecs.shape == (2, svc.dim)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_add_and_search_vectors(svc):
    svc.add_vectors([1, 2, 3], ["cat", "dog", "fish"])
    results = svc.search_vectors("cat", k=2)
    assert len(results) > 0
    assert all("chunk_id" in r and "score" in r for r in results)


def test_remove_vectors(svc):
    svc.add_vectors([10, 20], ["apple", "banana"])
    assert svc.index.ntotal == 2
    svc.remove_vectors([10])
    assert svc.index.ntotal == 1


def test_search_empty_index_returns_empty(svc):
    results = svc.search_vectors("anything", k=5)
    assert results == []


def test_index_persists_across_reload(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    index_path = str(tmp_path / "index.faiss")
    svc1 = VectorService(index_path=index_path, db=db)
    svc1.add_vectors([1, 2], ["persisted text", "another chunk"])
    # Reload
    svc2 = VectorService(index_path=index_path, db=db)
    assert svc2.index.ntotal == 2
