

# **SCALAR 2.0: Project Workflow**

This document provides a detailed, step-by-step explanation of the internal workflows of the SCALAR 2.0 application. It traces the journey of a request from the moment it is sent until a response is returned, covering both the document ingestion and search processes.

## **Workflow 1: The Ingestion Flow (Uploading a PDF)**

This workflow is initiated when a user uploads a PDF document to be added to the vector database.

### **Step 1: HTTP Request Reception (API Layer)**

-   **File:** `api/main.py`
-   **Function:** `upload_pdf`

1.  A client (such as the Streamlit UI or a `curl` command) sends an `HTTP POST` request with a `multipart/form-data` payload to the `/upload` endpoint.
2.  The FastAPI server receives the request. It validates that the uploaded file's `Content-Type` is `application/pdf`. If not, it rejects the request with a `400 Bad Request` error.
3.  The file is saved to a temporary directory (`./temp_uploads`) on the server's local disk. This is necessary so that the `PyMuPDF` library can access it via a file path.
4.  The API then calls the core processing method: `vector_service.process_and_store_pdf()`, passing the path to the temporary file and the original filename.

### **Step 2: Pre-processing and Text Extraction (Core Service Layer)**

-   **File:** `core/vector_service.py`
-   **Function:** `process_and_store_pdf()`

1.  **Duplicate Check:** The first action is to check if a document with the same filename has already been indexed. It iterates through the `self.metadata` list to see if the `source` key matches the incoming `filename`. If a match is found, it raises a `ValueError`, which is caught by the API layer and returned as a `422 Unprocessable Entity` error.
2.  **Text Extraction:** The `fitz` (PyMuPDF) library opens the PDF file. It iterates through each page, extracts the raw text, and checks if it contains content.
3.  **Data Structuring:** For each page with text, a LangChain `Document` object is created. This object holds the `page_content` (the text) and its associated `metadata` (the source filename and page number). This ensures the context of each piece of text is preserved. If no text is extracted from the entire PDF, a `ValueError` is raised.

### **Step 3: Chunking and Embedding (Core Service Layer)**

-   **File:** `core/vector_service.py`
-   **Function:** `process_and_store_pdf()`

1.  **Text Chunking:** The list of `Document` objects is passed to `LangChain's RecursiveCharacterTextSplitter`. This tool intelligently splits the text from all pages into smaller, overlapping chunks (of max 1000 characters with a 200-character overlap). It prioritizes splitting along paragraph breaks, then line breaks, then sentences, to keep the chunks as semantically coherent as possible. The output is a new list of smaller `Document` objects, each representing one chunk.
2.  **Data Separation:** The content and metadata from the chunks are separated into two lists: `chunk_texts` (a list of strings) and `chunk_metadata` (a list of dictionaries).
3.  **Vector Embedding:** The `chunk_texts` list is passed to the `SentenceTransformer` model's `encode()` method. This model transforms each text chunk into a 384-dimensional numerical vector (embedding). The result is a single NumPy array where each row is a vector.
4.  **Normalization:** The vectors in the NumPy array are normalized to a unit length (L2 normalization). This is a required step for performing accurate similarity searches using an inner product metric.

### **Step 4: Indexing and Persistence (Core Service Layer)**

-   **File:** `core/vector_service.py`
-   **Functions:** `process_and_store_pdf()`, `_save_data()`

1.  **Add to Index:** The NumPy array of vectors is added to the in-memory FAISS index (`self.index.add()`). FAISS automatically assigns an implicit, sequential ID (its position) to each vector.
2.  **Add to Lists:** The `chunk_metadata` and `chunk_texts` are extended to the service's master lists, `self.metadata` and `self.content`. The position of each item in these lists directly corresponds to the position of its vector in the FAISS index.
3.  **Save to Disk:** The `_save_data()` method is called. It uses `faiss.write_index()` to save the vector index to `index.faiss`, and `pickle.dump()` to serialize and save the `metadata` and `content` lists to `metadata.pkl` and `content.pkl`.

### **Step 5: Final Response (API Layer)**

-   **File:** `api/main.py`
-   **Function:** `upload_pdf()`

1.  Control returns to the API layer after the `vector_service` method completes successfully.
2.  The `finally` block in the `try...except` statement executes, deleting the file from the temporary directory.
3.  The API sends a `200 OK` response to the client with a JSON body confirming the successful ingestion.

---

## **Workflow 2: The Search Flow (Asking a Question)**

This workflow is initiated when a user submits a query to find relevant information within the indexed documents.

### **Step 1: HTTP Request Reception (API Layer)**

-   **File:** `api/main.py`
-   **Function:** `search()`

1.  A client sends an `HTTP POST` request to the `/search` endpoint with a JSON body containing the `query_text` and the desired number of results, `k`.
2.  FastAPI uses the `SearchQuery` Pydantic model to automatically validate the request body. If the data is invalid (e.g., `query_text` is missing), it returns a `422 Unprocessable Entity` error.
3.  The API calls the core search method: `vector_service.search()`, passing the query text and `k`.

### **Step 2: Query Processing and Vector Search (Core Service Layer)**

-   **File:** `core/vector_service.py`
-   **Function:** `search()`

1.  **Database Check:** The function first checks if the database contains any vectors (`self.index.ntotal == 0`). If not, it returns an error dictionary.
2.  **Query Embedding:** The user's `query_text` (a single string) is passed to the *same* `SentenceTransformer` model to convert it into a 384-dimensional vector. This ensures the query and the documents are in the same vector space. The query vector is also normalized.
3.  **FAISS Search:** The `self.index.search()` method is called. This is the core retrieval step. The FAISS HNSW algorithm efficiently navigates its internal graph structure to find the `k` vectors in the index that are most similar to the query vector.
4.  **Search Output:** The search returns two NumPy arrays:
    *   `distances`: An array of similarity scores (higher is more similar).
    *   `indices`: An array containing the integer positions (the implicit IDs) of the `k` most similar vectors in the index.

### **Step 3: Data Retrieval and Response Formatting (Core Service Layer)**

-   **File:** `core/vector_service.py`
-   **Function:** `search()`

1.  **Data Lookup:** The function iterates through the `indices` array returned by FAISS.
2.  For each index `idx`, it retrieves the corresponding metadata and content from the in-memory lists: `self.metadata[idx]` and `self.content[idx]`.
3.  **Response Packaging:** The retrieved metadata and content are combined into a structured dictionary for each result. The similarity score from the `distances` array is also included.
4.  All result dictionaries are collected into a final list, which is then wrapped in a parent dictionary: `{"results": [...]}`. This dictionary is returned to the API layer.

### **Step 4: Final Response (API Layer)**

-   **File:** `api/main.py`
-   **Function:** `search()`

1.  The API layer receives the dictionary of results from the `vector_service`.
2.  If the dictionary contains an "error" key, it raises an `HTTPException`.
3.  Otherwise, it sends a `200 OK` response to the client, with the structured list of search results as the JSON body.