import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from langchain_text_splitters import RecursiveCharacterTextSplitter
from api.main import db, vector_svc, bm25_svc
from core.parsers import parse_file

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
CHUNK_SIZE = int(os.getenv("SCALAR_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("SCALAR_CHUNK_OVERLAP", "200"))


@router.get("/")
def list_documents(namespace: str = Query("default")):
    docs = db.list_sources(namespace)
    return {"namespace": namespace, "documents": docs}


@router.delete("/{source}")
def delete_document(source: str, namespace: str = Query("default")):
    if not db.source_exists(source, namespace):
        raise HTTPException(
            status_code=404,
            detail=f"'{source}' not found in namespace '{namespace}'."
        )
    # Get chunk IDs before soft-delete to remove from FAISS
    active_ids = [
        c["id"] for c in db.get_active_chunks(namespace)
        if c["source"] == source
    ]
    db.soft_delete_source(source, namespace)
    if active_ids:
        vector_svc.remove_vectors(active_ids)
    bm25_svc.rebuild(db.get_active_chunks(namespace))
    return {"status": "deleted", "source": source, "namespace": namespace}


@router.put("/{source}")
async def update_document(
    source: str,
    file: UploadFile = File(...),
    namespace: str = Query("default"),
):
    """Delete existing chunks + re-ingest new file under the same source name."""
    if db.source_exists(source, namespace):
        active_ids = [
            c["id"] for c in db.get_active_chunks(namespace)
            if c["source"] == source
        ]
        db.soft_delete_source(source, namespace)
        if active_ids:
            vector_svc.remove_vectors(active_ids)

    tmp_path = os.path.join(TEMP_DIR, f"update_{namespace}_{file.filename}")
    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        pages = parse_file(tmp_path, file.filename)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        rows = []
        for page in pages:
            for i, chunk_text in enumerate(splitter.split_text(page["text"])):
                rows.append({
                    "source": source,
                    "namespace": namespace,
                    "page_number": page["page_number"],
                    "chunk_index": i,
                    "content": chunk_text,
                })

        if not rows:
            raise HTTPException(status_code=422, detail="No chunks extracted.")

        chunk_ids = db.insert_chunks_batch(rows)
        vector_svc.add_vectors(chunk_ids, [r["content"] for r in rows])
        bm25_svc.rebuild(db.get_active_chunks(namespace))

    except HTTPException:
        raise
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        await file.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "status": "updated",
        "source": source,
        "namespace": namespace,
        "chunks_indexed": len(rows),
    }
