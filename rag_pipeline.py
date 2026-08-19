import os
from typing import List, Any, Callable, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import pymupdf as fitz
import numpy as np

from langchain_core.documents import Document

PERSIST_DIRECTORY = "./chroma_db"

# Shared singleton OCR engine
_SHARED_OCR = None

def get_ocr():
    global _SHARED_OCR
    if _SHARED_OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _SHARED_OCR = RapidOCR()
    return _SHARED_OCR

def get_embedding_function():
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

def _process_single_page(args: tuple) -> tuple:
    page_idx, pdf_bytes, filename, enable_ocr = args
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]
    
    # 1. Instant digital text check
    text = page.get_text().strip()
    
    # 2. Fast OCR if no digital text
    if enable_ocr and (not text or len(text) < 20):
        ocr = get_ocr()
        pix = page.get_pixmap(dpi=80, colorspace=fitz.csGRAY)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        ocr_results, _ = ocr(img_array)
        if ocr_results:
            text = " ".join([line[1] for line in ocr_results]).strip()
            
    doc.close()
    return page_idx, text, filename

def extract_documents_from_uploaded_files(
    uploaded_files: List[Any],
    start_page: int = 1,
    end_page: Optional[int] = None,
    enable_ocr: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> List[Document]:
    documents: List[Document] = []
    
    # Pre-warm OCR engine once before threads spawn
    if enable_ocr:
        get_ocr()

    max_workers = min(6, os.cpu_count() or 4)

    for file_idx, file in enumerate(uploaded_files):
        filename = file.name
        file.seek(0)
        pdf_bytes = file.read()
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()

        actual_start = max(0, start_page - 1)
        actual_end = min(total_pages, end_page) if end_page else total_pages
        target_pages = list(range(actual_start, actual_end))
        pages_to_process = len(target_pages)

        tasks = [(idx, pdf_bytes, filename, enable_ocr) for idx in target_pages]
        completed_pages = 0
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(_process_single_page, task): task for task in tasks}
            
            for future in as_completed(future_to_page):
                page_idx, text, fname = future.result()
                completed_pages += 1
                
                if progress_callback:
                    extract_ratio = completed_pages / max(1, pages_to_process)
                    progress_value = extract_ratio * 0.80
                    progress_callback(
                        f"Reading '{fname}' — Page {page_idx + 1} ({completed_pages}/{pages_to_process})",
                        progress_value,
                    )
                
                if text:
                    results.append((page_idx, text, fname))

        results.sort(key=lambda x: x[0])
        for page_idx, text, fname in results:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": fname, "page": page_idx + 1},
                )
            )

    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from config import CHUNK_SIZE, CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)

def create_vector_store(chunks: List[Document], embeddings: Any):
    from langchain_chroma import Chroma
    from config import COLLECTION_NAME
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Passage {i} | Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})]\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs, 1)
    )

RAG_PROMPT_TEMPLATE = """You are a helpful AI research assistant.
Answer the question using strictly the retrieved context below.
If the context does not contain the answer, say "I cannot find the answer based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

def build_rag_chain(
    vector_store: Any,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
):
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from config import OLLAMA_MODEL, TEMPERATURE, TOP_K

    model_name = model_name or OLLAMA_MODEL
    temperature = temperature if temperature is not None else TEMPERATURE
    top_k = top_k or TOP_K

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_generator = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_generator)
