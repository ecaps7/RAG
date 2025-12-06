import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "database"

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database paths
SQL_DB_PATH = os.getenv("SQL_DB_PATH", str(DATA_DIR / "financial_rag.db"))
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", str(DATA_DIR / "financial_vectors.db"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(DATA_DIR / "bm25_index.pkl"))

# Embedding model configurations
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 4096))
MILVUS_COLLECTION = "financial_chunks"

# Reranker configurations (Ollama)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "dengcao/Qwen3-Reranker-8B:Q5_K_M")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RERANKER_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", 0.0))  # Qwen3-Reranker 输出 0-1 之间的分数

# RRF configurations
RRF_K = int(os.getenv("RRF_K", 60))  # RRF 常数，通常取 60