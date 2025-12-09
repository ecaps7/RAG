"""Application settings and model initialization."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

# Try different import paths for init_chat_model
try:
    from langchain_core.language_models import init_chat_model
except ImportError:
    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        init_chat_model = None  # type: ignore

from .constants import SOURCE_RELIABILITY, TOP_K


# Load environment variables
load_dotenv()


@dataclass
class AppConfig:
    """Application configuration loaded from environment variables."""
    
    # ===== Model Settings =====
    response_model_name: str = os.getenv("RESPONSE_MODEL", "deepseek:deepseek-chat")
    response_model_temperature: float = float(os.getenv("RESPONSE_TEMPERATURE", "0.7"))

    # ===== API Keys / Base URLs =====
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    ark_api_key: str | None = os.getenv("ARK_API_KEY")
    ark_base_url: str | None = os.getenv("ARK_BASE_URL")

    # ===== Project Structure =====
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "database"

    # ===== Retrieval & Vector Store =====
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "outputs/vector_store")
    all_chunks_path: str = os.getenv("ALL_CHUNKS_PATH", "outputs/all_chunks.jsonl")
    rebuild_vector_store: bool = os.getenv("REBUILD_VECTOR_STORE", "false").lower() == "true"

    # ===== Database Paths =====
    sql_db_path: str = os.getenv("SQL_DB_PATH", str(data_dir / "financial_rag.db"))
    milvus_db_path: str = os.getenv("MILVUS_DB_PATH", str(data_dir / "financial_vectors.db"))
    bm25_index_path: str = os.getenv("BM25_INDEX_PATH", str(data_dir / "bm25_index.pkl"))
    
    # Ollama embedding model (run locally via Ollama)
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Legacy HuggingFace model config (kept for backward compatibility)
    hf_model: str = os.getenv("HF_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    
    # Embedding backend: "ollama" or "huggingface"
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "ollama")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "2560"))
    
    # Milvus settings
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "financial_chunks")
    
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "8"))
    
    # ===== MMR Settings =====
    use_mmr: bool = os.getenv("USE_MMR", "true").lower() == "true"
    mmr_lambda_mult: float = float(os.getenv("MMR_LAMBDA_MULT", "0.3"))
    mmr_fetch_multiplier: float = float(os.getenv("MMR_FETCH_MULTIPLIER", "3.0"))

    # ===== Reranker Settings =====
    reranker_model: str = os.getenv("RERANKER_MODEL", str(project_root / "model" / "Qwen3-Reranker-0.6B"))
    reranker_threshold: float = float(os.getenv("RERANKER_THRESHOLD", "0.0"))  # Qwen3-Reranker outputs scores between 0-1

    
    # ===== RRF Settings =====
    rrf_k: int = int(os.getenv("RRF_K", "60"))  # RRF constant, usually set to 60


# Singleton config instance
_CONFIG_INSTANCE: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the application configuration (singleton)."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = AppConfig()
    return _CONFIG_INSTANCE


def init_model_with_config(model_name: str, temperature: float | None = None):
    """Initialize a chat model based on environment configuration.
    
    Supports Doubao/DeepSeek/Google/Ollama providers.
    """
    cfg = get_config()

    api_key = None
    base_url = None
    model_provider = None

    # Route to ARK for Doubao models
    if "doubao" in model_name.lower():
        api_key = cfg.ark_api_key
        base_url = cfg.ark_base_url
    elif "ollama" in model_name.lower() or model_name in ["qwen3:14b", "qwen3-embedding:4b"]:
        model_provider = "ollama"
        base_url = cfg.ollama_base_url
    else:
        # Default to DeepSeek; fallback to Google
        api_key = cfg.deepseek_api_key or cfg.google_api_key

    params: Dict[str, object] = {
        "api_key": api_key,
        "temperature": temperature if temperature is not None else cfg.response_model_temperature,
    }
    
    # 针对 Ollama 模型添加特定参数
    if model_provider == "ollama" and model_name == "qwen3:14b":
        # 关闭 qwen3:14b 的思考模式
        params["model_kwargs"] = {"reasoning": False, "keep_alive": False}
    if base_url:
        params["base_url"] = base_url
    if model_provider:
        params["model_provider"] = model_provider

    return init_chat_model(model_name, **params)
