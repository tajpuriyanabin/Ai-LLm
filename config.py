"""
Configuration constants for Offline Advanced RAG & Study Hub.
"""
# Local LLM (Ollama)
OLLAMA_MODEL = "llama3.2"
TEMPERATURE = 0.2

# Local Embeddings (CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Text Chunking Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Advanced Retrieval Settings
RETRIEVAL_CANDIDATES = 10  # Top candidates retrieved by Hybrid search
TOP_K = 3                  # Top reranked passages passed to LLM
BM25_WEIGHT = 0.4          # Keyword weight
VECTOR_WEIGHT = 0.6        # Semantic vector weight

# Storage Paths
COLLECTION_NAME = "offline_rag_docs"
PERSIST_DIRECTORY = "./chroma_db"
SESSIONS_FILE = "chat_sessions.json"
