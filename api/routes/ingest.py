import os
import shutil
import uuid
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, Query
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.main import db, vector_svc, bm25_svc
from core.parsers import parse_file

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

CHUNK_SIZE = int(os.getenv("SCALAR_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("SCALAR_CHUNK_OVERLAP", "200"))

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@router.post("/")
async def ingest_file(
    file: UploadFile = File(...),
    namespace: str = Query("default", description="Namespace to ingest into"),
):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    if db.source_exists(filename, namespace):
        raise HTTPException(
            status_code=422,
            detail=f"'{filename}' is already indexed in namespace '{namespace}'. "
                   f"DELETE it first or use a different filename."
        )

    tmp_path = os.path.join(TEMP_DIR, f"{namespace}_{filename}")
    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        pages = parse_file(tmp_path, filename)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )

        rows = []
        for page in pages:
            for i, chunk_text in enumerate(splitter.split_text(page["text"])):
                rows.append({
                    "source": filename,
                    "namespace": namespace,
                    "page_number": page["page_number"],
                    "chunk_index": i,
                    "content": chunk_text,
                })

        if not rows:
            raise HTTPException(status_code=422, detail="No text chunks could be extracted.")

        chunk_ids = db.insert_chunks_batch(rows)
        vector_svc.add_vectors(chunk_ids, [r["content"] for r in rows])
        bm25_svc.rebuild(db.get_active_chunks(namespace))

    except HTTPException:
        raise
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    finally:
        await file.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "status": "success",
        "source": filename,
        "namespace": namespace,
        "chunks_indexed": len(rows),
    }


def _run_ingest_job(job_id: str, tmp_path: str, filename: str, namespace: str):
    """Background task: process file, update job status in DB."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter as Splitter
        pages = parse_file(tmp_path, filename)
        splitter = Splitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        rows = []
        for page in pages:
            for i, chunk_text in enumerate(splitter.split_text(page["text"])):
                rows.append({
                    "source": filename,
                    "namespace": namespace,
                    "page_number": page["page_number"],
                    "chunk_index": i,
                    "content": chunk_text,
                })
        chunk_ids = db.insert_chunks_batch(rows)
        vector_svc.add_vectors(chunk_ids, [r["content"] for r in rows])
        bm25_svc.rebuild(db.get_active_chunks(namespace))
        db.update_job(job_id, "completed", f"Indexed {len(rows)} chunks")
    except Exception as e:
        db.update_job(job_id, "failed", str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/async")
async def ingest_file_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Query("default"),
):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")
    if db.source_exists(filename, namespace):
        raise HTTPException(status_code=422, detail=f"'{filename}' already indexed.")

    job_id = str(uuid.uuid4())
    tmp_path = os.path.join(TEMP_DIR, f"async_{job_id}_{filename}")
    with open(tmp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    await file.close()

    db.create_job(job_id, filename, namespace)
    background_tasks.add_task(_run_ingest_job, job_id, tmp_path, filename, namespace)

    return {"job_id": job_id, "status": "pending", "source": filename, "namespace": namespace}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
