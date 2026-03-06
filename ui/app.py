import io
import streamlit as st
import requests

st.set_page_config(page_title="SCALAR Vector DB", page_icon="🔷", layout="wide")
st.title("🔷 SCALAR Vector Database")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API Base URL", value="http://127.0.0.1:8000")
    api_key = st.text_input("API Key", type="password", value="")
    namespace = st.text_input("Namespace", value="default")
    st.divider()
    if st.button("📊 View Stats"):
        try:
            r = requests.get(f"{api_url}/stats", headers={"X-API-Key": api_key}, timeout=10)
            if r.status_code == 200:
                st.json(r.json())
            else:
                st.error(f"Stats error: {r.status_code}")
        except Exception as e:
            st.error(str(e))

HEADERS = {"X-API-Key": api_key}

tab1, tab2, tab3 = st.tabs(["📤 Ingest", "🔍 Search", "📄 Documents"])

# --- Tab 1: Ingest ---
with tab1:
    st.header("Ingest Documents")
    st.markdown("Upload PDF, DOCX, TXT, or Markdown files to build your knowledge base.")
    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            with st.spinner(f"Indexing **{f.name}**..."):
                try:
                    resp = requests.post(
                        f"{api_url}/ingest/",
                        params={"namespace": namespace},
                        files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                        headers=HEADERS,
                        timeout=600,
                    )
                    if resp.status_code == 200:
                        d = resp.json()
                        st.success(f"✅ **{f.name}** — {d['chunks_indexed']} chunks indexed")
                    else:
                        st.error(f"❌ **{f.name}**: {resp.json().get('detail', resp.status_code)}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# --- Tab 2: Search ---
with tab2:
    st.header("Search")
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        query = st.text_input("Query", placeholder="What is machine learning?")
    with col2:
        k = st.number_input("Top K", min_value=1, max_value=100, value=5)
    with col3:
        rerank = st.checkbox("Rerank")

    if st.button("Search 🔍"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                try:
                    resp = requests.post(
                        f"{api_url}/search/",
                        json={
                            "query_text": query,
                            "k": k,
                            "namespace": namespace,
                            "rerank": rerank,
                        },
                        headers=HEADERS,
                        timeout=120,
                    )
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    st.stop()

            if resp.status_code == 200:
                results = resp.json()["results"]
                if not results:
                    st.info("No results found.")
                else:
                    st.success(f"Found {len(results)} result(s)")
                    for i, r in enumerate(results):
                        page = r.get("page_number", "?")
                        with st.expander(
                            f"**#{i+1}** | {r['source']} · page {page} · score {r['score']:.4f}"
                        ):
                            st.markdown(r["content"])
            else:
                st.error(f"Search failed: {resp.json().get('detail', resp.status_code)}")

# --- Tab 3: Documents ---
with tab3:
    st.header("Manage Documents")
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        refresh = st.button("🔄 Refresh List")

    if refresh:
        try:
            resp = requests.get(
                f"{api_url}/documents/",
                params={"namespace": namespace},
                headers=HEADERS,
                timeout=10,
            )
            if resp.status_code == 200:
                docs = resp.json()["documents"]
                if not docs:
                    st.info(f"No documents in namespace **{namespace}**.")
                else:
                    st.write(f"**{len(docs)}** document(s) in namespace `{namespace}`:")
                    for doc in docs:
                        col1, col2, col3 = st.columns([5, 1, 1])
                        col1.write(f"📄 **{doc['source']}** — {doc['chunk_count']} chunks")
                        col2.write(f"`{doc.get('created_at', '')[:10]}`")
                        if col3.button("🗑 Delete", key=f"del_{doc['source']}"):
                            del_resp = requests.delete(
                                f"{api_url}/documents/{doc['source']}",
                                params={"namespace": namespace},
                                headers=HEADERS,
                                timeout=30,
                            )
                            if del_resp.status_code == 200:
                                st.toast(f"Deleted {doc['source']}")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {del_resp.json().get('detail')}")
            else:
                st.error(f"Failed to list documents: {resp.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")
