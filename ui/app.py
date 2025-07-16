import streamlit as st
import requests
import os

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"  # URL of your FastAPI backend

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("📄🔍 Semantic PDF Search Engine")
st.markdown("""
This application allows you to build a searchable knowledge base from your PDF documents.
- **Step 1:** Upload one or more PDF files. The system will process and index their content.
- **Step 2:** Ask a question or enter keywords to search across all uploaded documents.
""")

# --- Helper Functions to Interact with API ---

def upload_document(file_path):
    """Sends a file to the /upload endpoint."""
    try:
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=600)
            return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to API: {e}"}

def perform_search(query, k):
    """Sends a query to the /search endpoint."""
    payload = {"query_text": query, "k": k}
    try:
        response = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get('detail', 'Unknown error')}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to API: {e}"}


# --- UI Components ---

# 1. Ingestion Pipeline Section
st.header("Step 1: Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF files to add to the knowledge base",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # To handle the file, we need to save it temporarily
        temp_dir = "temp_ui_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
            # We call a slightly modified function to get more error details
            try:
                with open(temp_path, "rb") as f_rb:
                    files = {'file': (os.path.basename(temp_path), f_rb, 'application/pdf')}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=600)
                
                # Check for non-200 status codes
                if response.status_code != 200:
                    # Try to get the detailed error from the JSON response
                    error_detail = response.json().get('detail', 'No detailed error message provided.')
                    st.error(f"Error processing {uploaded_file.name}: {error_detail}")
                else:
                    st.success(f"✅ Successfully indexed '{uploaded_file.name}'!")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to API while processing {uploaded_file.name}: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred with {uploaded_file.name}: {e}")

        # Clean up the temporary file
        os.remove(temp_path)


# 2. Query Engine Section
st.header("Step 2: Search the Knowledge Base")

col1, col2 = st.columns([8, 1])
with col1:
    query = st.text_input("Enter your search query:", placeholder="e.g., What is the main idea of the document?")
with col2:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=3)

if st.button("Search 🔍"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            search_results = perform_search(query, top_k)

        # --- Find this section at the bottom of ui/app.py ---

        st.subheader("Search Results")
        if "error" in search_results:
            st.error(f"Search failed: {search_results['error']}")
        elif not search_results.get("results"):
            st.info("No relevant results found.")
        else:
            for i, result in enumerate(search_results["results"]):
                # --- NEW: Displaying the page number from the metadata ---
                metadata = result['metadata']
                source = metadata.get('source', 'Unknown Source')
                page_num = metadata.get('page_number', 'N/A')
                
                expander_title = (
                    f"**Result {i+1}** | "
                    f"Source: `{source}` | "
                    f"Page: `{page_num}` | "
                    f"Similarity: {result['similarity']:.2f}"
                )
                
                with st.expander(expander_title):
                    st.markdown(metadata['content'])