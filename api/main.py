# main.py

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# This adds the parent directory to the system path to allow for imports
# from the 'core' module where VectorService is located.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_service import VectorService

# --- API Application Setup ---
app = FastAPI(
    title="Vector Database Service API",
    description="An API for ingesting PDF documents and performing semantic search.",
    version="1.0.0"
)

# --- Singleton Service Instantiation ---
# The VectorService class is instantiated once when the application starts.
# This single instance handles all business logic and is shared across all API requests,
# which is efficient as the model and data are only loaded into memory once.
vector_service = VectorService(data_path="data_store")

# --- Configuration for Temporary File Storage ---
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


# --- Pydantic Data Models ---
# These models define the expected data structure for API requests and responses.
# They provide automatic data validation and generate API documentation.

class SearchQuery(BaseModel):
    query_text: str = Field(..., min_length=1, description="The text to search for.")
    k: int = Field(5, gt=0, le=20, description="The number of results to return.")

class SearchResult(BaseModel):
    metadata: Dict[str, Any]
    similarity: float

class SearchResponse(BaseModel):
    results: List[SearchResult]


# --- API Endpoints ---

@app.post("/upload", summary="Upload and process a PDF document")
async def upload_pdf(file: UploadFile = File(..., description="The PDF file to be processed.")):
    """
    Handles the ingestion of a PDF file. The file is saved temporarily,
    processed by the VectorService, and then the temporary file is deleted.

    Args:
        file (UploadFile): The PDF file uploaded by the client.

    Returns:
        Dict[str, str]: A JSON object confirming the successful ingestion.
                        Example: {"status": "success", "filename": "example.pdf", ...}

    Raises:
        HTTPException:
            - 400: If the uploaded file is not of content-type 'application/pdf'.
            - 422: If the service cannot process the file (e.g., it's a duplicate or contains no text).
            - 500: For any other unexpected server errors during processing.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

    try:
        # Save the file to a temporary location on disk.
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Pass the file path to the core service for the main ingestion logic.
        vector_service.process_and_store_pdf(pdf_file_path=temp_file_path, filename=file.filename)

    except ValueError as e:
        # Catch specific processing errors raised by the service.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Catch all other potential exceptions.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # This block ensures the temporary file is deleted after processing,
        # regardless of whether an error occurred.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await file.close()

    return {"status": "success", "filename": file.filename, "detail": "File processed and indexed successfully."}


@app.post("/search", response_model=SearchResponse, summary="Perform a semantic search")
async def search(query: SearchQuery = Body(...)):
    """
    Performs a semantic search based on a user's query text.

    Args:
        query (SearchQuery): A request body containing the 'query_text' and the
                             number of results to return, 'k'.
                             Example: {"query_text": "What is machine learning?", "k": 3}

    Returns:
        SearchResponse: A JSON object containing a list of the top k search results.
                        Example: {"results": [{"metadata": {...}, "similarity": 0.85}, ...]}

    Raises:
        HTTPException:
            - 400: If the search is attempted on an empty database.
    """
    search_results = vector_service.search(query_text=query.query_text, k=query.k)
    if "error" in search_results:
            raise HTTPException(status_code=400, detail=search_results["error"])
    return search_results


@app.get("/", summary="Root endpoint for health check")
def read_root():
    """
    Provides a simple health check endpoint to confirm that the API server is running.

    Returns:
        Dict[str, str]: A message indicating the API status.
    """
    return {"status": "Vector Database API is running."}