import io
import os
import zipfile
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.main import db, vector_svc

router = APIRouter()

DATA_DIR = os.getenv("SCALAR_DATA_DIR", "data_store")


@router.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@router.get("/stats")
def get_stats():
    stats = db.get_stats()
    stats["index_vector_count"] = vector_svc.index.ntotal
    stats["embedding_model"] = os.getenv("SCALAR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    stats["index_type"] = "IndexIDMap(IndexFlatIP)"
    return stats


@router.get("/admin/backup")
def backup():
    """Stream a zip of the data_store directory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(DATA_DIR):
            for fname in os.listdir(DATA_DIR):
                full = os.path.join(DATA_DIR, fname)
                if os.path.isfile(full):
                    zf.write(full, fname)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=scalar_backup.zip"},
    )
