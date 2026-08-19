import streamlit as st
import pymupdf as fitz
import config
from rag_pipeline import (
    get_embedding_function,
    extract_documents_from_uploaded_files,
    chunk_documents,
    create_vector_store,
    build_rag_chain,
)

# Page configuration
st.set_page_config(page_title="Offline Document RAG", page_icon="🔒", layout="wide")

# State initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "indexed_files_info" not in st.session_state:
    st.session_state.indexed_files_info = None

# Cached embedding loader (Loads only on demand, NOT on startup)
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return get_embedding_function()

# Sidebar
with st.sidebar:
    st.title("🔒 Offline RAG Setup")
    st.success(f"Running locally with **{config.OLLAMA_MODEL}** (Ollama)")
    
    st.markdown("---")
    st.subheader("📄 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    start_page = 1
    end_page = None
    enable_ocr = True

    if uploaded_files:
        try:
            uploaded_files[0].seek(0)
            temp_doc = fitz.open(stream=uploaded_files[0].read(), filetype="pdf")
            total_pdf_pages = len(temp_doc)
            temp_doc.close()
            uploaded_files[0].seek(0)

            st.caption(f"Total Pages Detected: **{total_pdf_pages}**")

            with st.expander("⚙️ Speed & Page Range Controls", expanded=True):
                enable_ocr = st.checkbox("Enable OCR (for Scanned Pages)", value=True, help="Uncheck if your PDF already has selectable digital text for instant 1-second extraction.")
                
                process_all = st.checkbox("Process Entire PDF", value=False)
                if not process_all:
                    max_range = min(30, total_pdf_pages)
                    page_range = st.slider(
                        "Select Page Range to Index:",
                        min_value=1,
                        max_value=total_pdf_pages,
                        value=(1, max_range),
                        help="Select a range of pages to process quickly (e.g. 1 to 25)."
                    )
                    start_page, end_page = page_range
                else:
                    start_page, end_page = 1, total_pdf_pages
        except Exception:
            pass

    if st.button("🚀 Process & Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            progress_bar = st.progress(0, text="Starting document processing (0%)...")

            def update_progress(msg: str, progress: float):
                clamped_progress = min(max(progress, 0.0), 1.0)
                percent = int(clamped_progress * 100)
                progress_bar.progress(clamped_progress, text=f"📊 {percent}% — {msg}")

            try:
                # 1. Extract text from PDF
                raw_docs = extract_documents_from_uploaded_files(
                    uploaded_files,
                    start_page=start_page,
                    end_page=end_page,
                    enable_ocr=enable_ocr,
                    progress_callback=update_progress,
                )
                
                if not raw_docs:
                    progress_bar.empty()
                    st.error("No extractable text found in the selected pages.")
                else:
                    # 2. Chunk text
                    update_progress("Splitting text into semantic chunks...", 0.82)
                    chunks = chunk_documents(raw_docs)
                    
                    # 3. Load embedding model only when needed
                    update_progress("Initializing embedding engine...", 0.88)
                    embeddings = load_embeddings()

                    # 4. Generate vectors & store in Chroma
                    update_progress(f"Generating embeddings for {len(chunks)} chunks...", 0.94)
                    vector_store = create_vector_store(chunks, embeddings)
                    
                    # 5. Build chain
                    update_progress("Finalizing RAG pipeline...", 0.98)
                    chain = build_rag_chain(vector_store)
                    
                    st.session_state.rag_chain = chain
                    st.session_state.indexed_files_info = {
                        "file_count": len(uploaded_files),
                        "chunk_count": len(chunks),
                        "page_range": f"Pages {start_page} to {end_page}" if end_page else "All pages",
                        "file_names": [f.name for f in uploaded_files],
                    }
                    progress_bar.progress(1.0, text="✅ 100% — Indexing Complete!")
                    st.success(f"Indexed {len(chunks)} chunks ({st.session_state.indexed_files_info['page_range']}) successfully!")
            except Exception as e:
                progress_bar.empty()
                st.error(f"Error: {str(e)}")

    if st.session_state.indexed_files_info:
        st.markdown("---")
        st.markdown("### 📊 Index Status")
        st.info(
            f"**Files:** {st.session_state.indexed_files_info['file_count']}\n\n"
            f"**Indexed Range:** {st.session_state.indexed_files_info.get('page_range', 'All')}\n\n"
            f"**Total Chunks:** {st.session_state.indexed_files_info['chunk_count']}\n\n"
            + "\n".join([f"- {name}" for name in st.session_state.indexed_files_info['file_names']])
        )

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🔒 Offline RAG Assistant (Local LLM + ChromaDB)")
st.caption("100% offline, privacy-first question answering over your documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {idx}:** {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
                    st.text(doc.page_content)
                    if idx < len(message["sources"]):
                        st.divider()

user_query = st.chat_input("Ask a question about the uploaded documents...")

if user_query:
    if st.session_state.rag_chain is None:
        st.warning("Please upload and index your documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer locally..."):
                try:
                    result = st.session_state.rag_chain.invoke(user_query)
                    answer = result.get("answer", "")
                    retrieved_docs = result.get("context", [])

                    st.markdown(answer)

                    if retrieved_docs:
                        with st.expander("🔍 View Retrieved Sources"):
                            for idx, doc in enumerate(retrieved_docs, 1):
                                st.markdown(f"**Source {idx}:** {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
                                st.text(doc.page_content)
                                if idx < len(retrieved_docs):
                                    st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_docs,
                    })
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")
