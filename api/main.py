import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from api.logging_config import setup_logging, RequestLoggingMiddleware
setup_logging()

from fastapi import FastAPI
from api.auth import APIKeyMiddleware
from core.database import Database
from core.vector_service import VectorService
from core.bm25_service import BM25Service

# --- Storage paths ---
DATA_DIR = os.getenv("SCALAR_DATA_DIR", "data_store")
DB_PATH = os.path.join(DATA_DIR, "scalar.db")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Shared service singletons (module-level) ---
db = Database(DB_PATH)
vector_svc = VectorService(index_path=INDEX_PATH, db=db)
bm25_svc = BM25Service()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rebuild BM25 from all active chunks on startup
    all_chunks = db.list_all_sources()
    if all_chunks:
        # Get all active chunks across all namespaces for BM25 rebuild
        conn_chunks = []
        seen_ns = set()
        for src in all_chunks:
            ns = src["namespace"]
            if ns not in seen_ns:
                seen_ns.add(ns)
                conn_chunks.extend(db.get_active_chunks(ns))
        bm25_svc.rebuild(conn_chunks)
    yield


app = FastAPI(
    title="SCALAR Vector Database API",
    description="Production-grade vector database with hybrid search.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyMiddleware)


from api.routes import ingest as ingest_routes
app.include_router(ingest_routes.router, prefix="/ingest", tags=["Ingest"])

from api.routes import search as search_routes
app.include_router(search_routes.router, prefix="/search", tags=["Search"])

from api.routes import documents as documents_routes
app.include_router(documents_routes.router, prefix="/documents", tags=["Documents"])

from api.routes import admin as admin_routes
app.include_router(admin_routes.router, tags=["Admin"])
