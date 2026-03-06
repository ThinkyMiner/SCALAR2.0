import pytest
from core.database import Database


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


def test_insert_chunk_returns_id(db):
    chunk_id = db.insert_chunk(
        source="doc.pdf", namespace="default",
        page_number=1, chunk_index=0, content="Hello world"
    )
    assert isinstance(chunk_id, int)
    assert chunk_id > 0


def test_get_chunk_by_id(db):
    chunk_id = db.insert_chunk(
        source="doc.pdf", namespace="default",
        page_number=1, chunk_index=0, content="Hello world"
    )
    chunk = db.get_chunk(chunk_id)
    assert chunk["content"] == "Hello world"
    assert chunk["source"] == "doc.pdf"
    assert chunk["namespace"] == "default"
    assert chunk["deleted"] == 0


def test_get_chunks_by_ids(db):
    id1 = db.insert_chunk("a.pdf", "ns", 1, 0, "chunk one")
    id2 = db.insert_chunk("a.pdf", "ns", 1, 1, "chunk two")
    chunks = db.get_chunks_by_ids([id1, id2])
    assert len(chunks) == 2
    contents = {c["content"] for c in chunks}
    assert contents == {"chunk one", "chunk two"}


def test_soft_delete_marks_deleted(db):
    chunk_id = db.insert_chunk("doc.pdf", "default", 1, 0, "text")
    db.soft_delete_source("doc.pdf", "default")
    chunk = db.get_chunk(chunk_id)
    assert chunk["deleted"] == 1


def test_list_sources_excludes_deleted(db):
    db.insert_chunk("keep.pdf", "default", 1, 0, "text a")
    db.insert_chunk("del.pdf", "default", 1, 0, "text b")
    db.soft_delete_source("del.pdf", "default")
    sources = db.list_sources("default")
    names = [s["source"] for s in sources]
    assert "keep.pdf" in names
    assert "del.pdf" not in names


def test_get_all_active_chunks_for_namespace(db):
    db.insert_chunk("a.pdf", "ns1", 1, 0, "aaa")
    db.insert_chunk("b.pdf", "ns1", 1, 0, "bbb")
    db.insert_chunk("c.pdf", "ns2", 1, 0, "ccc")
    db.soft_delete_source("b.pdf", "ns1")
    chunks = db.get_active_chunks("ns1")
    contents = [c["content"] for c in chunks]
    assert "aaa" in contents
    assert "bbb" not in contents
    assert "ccc" not in contents


def test_get_stats(db):
    db.insert_chunk("a.pdf", "default", 1, 0, "text")
    db.insert_chunk("b.pdf", "default", 1, 0, "text2")
    stats = db.get_stats()
    assert stats["total_chunks"] >= 2
    assert stats["total_sources"] >= 2


def test_filtered_ids_for_namespace(db):
    id1 = db.insert_chunk("a.pdf", "ns1", 1, 0, "one")
    id2 = db.insert_chunk("b.pdf", "ns2", 1, 0, "two")
    active_ids = db.get_active_chunk_ids("ns1")
    assert id1 in active_ids
    assert id2 not in active_ids


def test_insert_chunks_batch(db):
    rows = [
        {"source": "batch.pdf", "namespace": "default", "page_number": 1, "chunk_index": 0, "content": "first"},
        {"source": "batch.pdf", "namespace": "default", "page_number": 1, "chunk_index": 1, "content": "second"},
    ]
    ids = db.insert_chunks_batch(rows)
    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)


def test_source_exists(db):
    db.insert_chunk("exists.pdf", "default", 1, 0, "text")
    assert db.source_exists("exists.pdf", "default") is True
    assert db.source_exists("nope.pdf", "default") is False


def test_source_exists_false_after_delete(db):
    db.insert_chunk("del.pdf", "default", 1, 0, "text")
    db.soft_delete_source("del.pdf", "default")
    assert db.source_exists("del.pdf", "default") is False
