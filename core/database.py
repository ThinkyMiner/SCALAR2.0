import sqlite3
import threading
from typing import List, Dict, Any, Optional


class Database:
    """SQLite-backed metadata store. Thread-safe via lock + WAL mode."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        with self._lock:
            conn = self._connect()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    source      TEXT    NOT NULL,
                    namespace   TEXT    NOT NULL DEFAULT 'default',
                    page_number INTEGER,
                    chunk_index INTEGER,
                    content     TEXT    NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted     INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_namespace
                    ON chunks(namespace, deleted);
                CREATE INDEX IF NOT EXISTS idx_chunks_source
                    ON chunks(source, namespace);
                CREATE TABLE IF NOT EXISTS jobs (
                    id         TEXT    PRIMARY KEY,
                    status     TEXT    NOT NULL DEFAULT 'pending',
                    source     TEXT,
                    namespace  TEXT,
                    result     TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            conn.close()

    def insert_chunk(self, source: str, namespace: str,
                     page_number: int, chunk_index: int,
                     content: str) -> int:
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "INSERT INTO chunks (source, namespace, page_number, chunk_index, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (source, namespace, page_number, chunk_index, content)
            )
            chunk_id = cur.lastrowid
            conn.commit()
            conn.close()
        return chunk_id

    def insert_chunks_batch(self, rows: List[Dict]) -> List[int]:
        """Insert many chunks atomically. rows: list of dicts with keys source, namespace, page_number, chunk_index, content."""
        with self._lock:
            conn = self._connect()
            ids = []
            for row in rows:
                cur = conn.execute(
                    "INSERT INTO chunks (source, namespace, page_number, chunk_index, content) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row["source"], row["namespace"], row["page_number"],
                     row["chunk_index"], row["content"])
                )
                ids.append(cur.lastrowid)
            conn.commit()
            conn.close()
        return ids

    def get_chunk(self, chunk_id: int) -> Optional[Dict]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_chunks_by_ids(self, ids: List[int]) -> List[Dict]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        conn = self._connect()
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND deleted = 0",
            ids
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_active_chunk_ids(self, namespace: str) -> List[int]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id FROM chunks WHERE namespace = ? AND deleted = 0",
            (namespace,)
        ).fetchall()
        conn.close()
        return [r["id"] for r in rows]

    def get_active_chunks(self, namespace: str) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE namespace = ? AND deleted = 0",
            (namespace,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def soft_delete_source(self, source: str, namespace: str) -> int:
        """Mark all chunks for a source as deleted. Returns affected row count."""
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "UPDATE chunks SET deleted = 1 WHERE source = ? AND namespace = ?",
                (source, namespace)
            )
            affected = cur.rowcount
            conn.commit()
            conn.close()
        return affected

    def source_exists(self, source: str, namespace: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM chunks WHERE source = ? AND namespace = ? AND deleted = 0 LIMIT 1",
            (source, namespace)
        ).fetchone()
        conn.close()
        return row is not None

    def list_sources(self, namespace: str) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute(
            """SELECT source, namespace,
                      COUNT(*) as chunk_count,
                      MIN(created_at) as created_at
               FROM chunks
               WHERE namespace = ? AND deleted = 0
               GROUP BY source, namespace
               ORDER BY MIN(created_at) DESC""",
            (namespace,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def list_all_sources(self) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute(
            """SELECT source, namespace,
                      COUNT(*) as chunk_count,
                      MIN(created_at) as created_at
               FROM chunks
               WHERE deleted = 0
               GROUP BY source, namespace
               ORDER BY namespace, MIN(created_at) DESC"""
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict:
        conn = self._connect()
        total_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE deleted = 0"
        ).fetchone()[0]
        total_sources = conn.execute(
            "SELECT COUNT(DISTINCT source || '|' || namespace) FROM chunks WHERE deleted = 0"
        ).fetchone()[0]
        namespaces = conn.execute(
            "SELECT DISTINCT namespace FROM chunks WHERE deleted = 0"
        ).fetchall()
        conn.close()
        return {
            "total_chunks": total_chunks,
            "total_sources": total_sources,
            "namespaces": [r[0] for r in namespaces],
        }

    # --- Job tracking ---

    def create_job(self, job_id: str, source: str, namespace: str):
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO jobs (id, status, source, namespace) VALUES (?, 'pending', ?, ?)",
                (job_id, source, namespace)
            )
            conn.commit()
            conn.close()

    def update_job(self, job_id: str, status: str, result: str = None):
        with self._lock:
            conn = self._connect()
            conn.execute(
                "UPDATE jobs SET status = ?, result = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, result, job_id)
            )
            conn.commit()
            conn.close()

    def get_job(self, job_id: str) -> Optional[Dict]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        return dict(row) if row else None
