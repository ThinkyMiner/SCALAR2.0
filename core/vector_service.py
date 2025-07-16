# vector_service.py

import os
import pickle
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF library for PDF processing
from langchain.schema import Document

class VectorService:
    """
    Encapsulates all vector-related operations, including data persistence,
    ingestion, indexing, and searching.
    """
    def __init__(self, data_path="data_store"):
        """
        Initializes the VectorService.

        This involves setting up file paths for data persistence, loading the
        embedding model, and loading existing data from disk if available.

        Args:
            data_path (str): The directory to store persistent data files
                             (index, metadata, content).
        """
        # Define paths for our three persistent data files.
        self.index_path = os.path.join(data_path, "index.faiss")
        self.metadata_path = os.path.join(data_path, "metadata.pkl")
        self.content_path = os.path.join(data_path, "content.pkl")
        os.makedirs(data_path, exist_ok=True)

        # Load the sentence-transformer model. This is a heavyweight object,
        # so it is only loaded once during the service's initialization.
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print("Model loaded.")

        # Initialize attributes to hold the database state in memory.
        self.index = None      # The FAISS index for fast vector search.
        self.metadata = []     # A list of metadata dicts for each chunk.
        self.content = []      # A list of the raw text content for each chunk.
        self._load_data()

    def _load_data(self):
        """Loads the index, metadata, and content from disk if all files exist."""
        if all(os.path.exists(p) for p in [self.index_path, self.metadata_path, self.content_path]):
            print("Loading existing data from persistence layer...")
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f_meta:
                    self.metadata = pickle.load(f_meta)
                with open(self.content_path, 'rb') as f_content:
                    self.content = pickle.load(f_content)
                print(f"Data loaded successfully. Index contains {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"Error loading data: {e}. Re-initializing empty index.")
                self._initialize_empty_index()
        else:
            print("No existing data found. Initializing a new index.")
            self._initialize_empty_index()

    def _initialize_empty_index(self):
        """Initializes a new, empty FAISS index with the HNSW algorithm."""
        print("Initializing new HNSW index.")
        # IndexHNSWFlat is a fast, approximate nearest neighbor search index.
        # M=64 sets the number of connections per node, a key performance parameter.
        # METRIC_INNER_PRODUCT is used for similarity with normalized vectors.
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, 64, faiss.METRIC_INNER_PRODUCT)
        self.metadata = []
        self.content = []

    def _save_data(self):
        """Saves the current index, metadata, and content lists to disk."""
        print("Saving data to persistence layer...")
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f_meta:
            pickle.dump(self.metadata, f_meta)
        with open(self.content_path, 'wb') as f_content:
            pickle.dump(self.content, f_content)
        print("Data saved.")

    def process_and_store_pdf(self, pdf_file_path: str, filename: str):
        """
        Processes a PDF file through the full ingestion pipeline.

        Args:
            pdf_file_path (str): The path to the PDF file on disk.
            filename (str): The original name of the file being processed.

        Returns:
            None: This method modifies the internal state of the service.

        Raises:
            ValueError: If the document has already been indexed or if no text
                        can be extracted from it.
        """
        # 1. Prevent duplicate processing.
        if any(meta.get('source') == filename for meta in self.metadata):
            raise ValueError(f"Document '{filename}' has already been processed and indexed.")

        # 2. Extract text from each page of the PDF.
        all_pages = []
        try:
            doc = fitz.open(pdf_file_path)
            for i, page in enumerate(doc):
                if page_text := page.get_text().strip():
                    all_pages.append(Document(
                        page_content=page_text,
                        metadata={"source": filename, "page_number": i + 1}
                    ))
            doc.close()
            if not all_pages:
                raise ValueError(f"No text could be extracted from the PDF '{filename}'.")
        except Exception as e:
            raise RuntimeError(f"Failed to read or process PDF '{filename}': {e}")

        # 3. Split the extracted text into semantically aware chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
        )
        chunks = text_splitter.split_documents(all_pages)
        
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_metadata = [chunk.metadata for chunk in chunks]

        # 4. Generate vector embeddings for each text chunk.
        embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=True)
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity search.

        # 5. Add new data to the in-memory stores.
        self.index.add(embeddings_np)
        self.metadata.extend(chunk_metadata)
        self.content.extend(chunk_texts)

        # 6. Persist the updated stores to disk.
        self._save_data()

    def search(self, query_text: str, k: int = 5):
        """
        Performs a semantic search on the indexed documents.

        Args:
            query_text (str): The user's natural language query.
            k (int): The number of top results to retrieve.

        Returns:
            Dict: A dictionary containing either the search results or an error message.
                  Example (Success): {"results": [...]}
                  Example (Failure): {"error": "Database is empty."}
        """
        if self.index.ntotal == 0:
            return {"error": "The vector database is empty. Please upload a document first."}

        # Set HNSW search-time parameter for a balance of speed and accuracy.
        self.index.hnsw.efSearch = 128

        # 1. Convert the query text into a vector.
        query_vector = self.embedding_model.encode([query_text])
        query_vector_np = np.array(query_vector).astype('float32')
        faiss.normalize_L2(query_vector_np)

        # 2. Perform the search using the FAISS index.
        # This returns the similarity scores and the indices (positions) of the matches.
        distances, indices = self.index.search(query_vector_np, k)

        # 3. Retrieve the original content and metadata using the returned indices.
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                result_metadata = self.metadata[idx].copy()
                result_metadata['content'] = self.content[idx]

                results.append({
                    "metadata": result_metadata,
                    "similarity": float(distances[0][i])
                })

        return {"results": results}