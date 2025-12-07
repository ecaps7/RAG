import os
from pathlib import Path

# 项目根目录（rag_agent 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "database"

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database paths
SQL_DB_PATH = os.getenv("SQL_DB_PATH", str(DATA_DIR / "financial_rag.db"))
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", str(DATA_DIR / "financial_vectors.db"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(DATA_DIR / "bm25_index.pkl"))

# Embedding model configurations
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 2560))
MILVUS_COLLECTION = "financial_chunks"

# Reranker configurations (HuggingFace - Local)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", str(PROJECT_ROOT / "model" / "Qwen3-Reranker-0.6B"))
RERANKER_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", 0.0))  # Qwen3-Reranker 输出 0-1 之间的分数

# RRF configurations
RRF_K = int(os.getenv("RRF_K", 60))  # RRF 常数，通常取 60