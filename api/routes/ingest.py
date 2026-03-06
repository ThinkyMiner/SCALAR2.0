import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
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
