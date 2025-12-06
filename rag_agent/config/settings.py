"""Application settings and model initialization."""

from __future__ import annotations

import os
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

    # ===== Retrieval & Vector Store =====
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "outputs/vector_store")
    all_chunks_path: str = os.getenv("ALL_CHUNKS_PATH", "outputs/all_chunks.jsonl")
    rebuild_vector_store: bool = os.getenv("REBUILD_VECTOR_STORE", "false").lower() == "true"
    
    # Ollama embedding model (run locally via Ollama)
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Legacy HuggingFace model config (kept for backward compatibility)
    hf_model: str = os.getenv("HF_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    
    # Embedding backend: "ollama" or "huggingface"
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "ollama")
    
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "8"))
    
    # ===== MMR Settings =====
    use_mmr: bool = os.getenv("USE_MMR", "true").lower() == "true"
    mmr_lambda_mult: float = float(os.getenv("MMR_LAMBDA_MULT", "0.3"))
    mmr_fetch_multiplier: float = float(os.getenv("MMR_FETCH_MULTIPLIER", "3.0"))
    
    # ===== BM25 Settings =====
    bm25_index_path: str = os.getenv(
        "BM25_INDEX_PATH",
        os.path.join(os.getenv("OUTPUTS_DIR", "outputs"), "bm25_index.json")
    )
    bm25_k1: float = float(os.getenv("BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("BM25_B", "0.75"))

    # ===== LLM Query Expansion =====
    use_llm_query_expansion: bool = os.getenv("USE_LLM_QUERY_EXPANSION", "true").lower() == "true"
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b")
    query_expansion_count: int = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))

    # ===== Chinese Stopwords =====
    zh_stopwords: set[str] = field(default_factory=lambda: {
        "的", "是", "在", "和", "与", "了", "等", "有", "为", "也", "我", "你", "他", "她",
        "及", "并", "对", "于", "或", "被", "其", "与其", "其中",
    })
    
    # ===== Source Reliability =====
    source_reliability: Dict[str, object] = field(default_factory=lambda: {
        "local": SOURCE_RELIABILITY.get("local", 0.9),
    })
    
    # ===== Top-K per Source =====
    top_k_per_source: Dict[str, int] = field(default_factory=lambda: TOP_K.copy())

    # ===== Fusion Layer Settings =====
    fusion_use_normalization: bool = os.getenv("FUSION_USE_NORMALIZATION", "true").lower() == "true"
    fusion_norm_method: str = os.getenv("FUSION_NORM_METHOD", "minmax")  # "minmax" | "zscore"
    fusion_norm_eps: float = float(os.getenv("FUSION_NORM_EPS", "1e-8"))
    fusion_use_mmr: bool = os.getenv("FUSION_USE_MMR", "true").lower() == "true"
    fusion_mmr_alpha: float = float(os.getenv("FUSION_MMR_ALPHA", "0.35"))
    fusion_mmr_fetch_multiplier: float = float(os.getenv("FUSION_MMR_FETCH_MULTIPLIER", "2.5"))


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
    
    Supports Doubao/DeepSeek/Google providers.
    """
    cfg = get_config()

    api_key = None
    base_url = None

    # Route to ARK for Doubao models
    if "doubao" in model_name.lower():
        api_key = cfg.ark_api_key
        base_url = cfg.ark_base_url
    else:
        # Default to DeepSeek; fallback to Google
        api_key = cfg.deepseek_api_key or cfg.google_api_key

    params: Dict[str, object] = {
        "api_key": api_key,
        "temperature": temperature if temperature is not None else cfg.response_model_temperature,
    }
    if base_url:
        params["base_url"] = base_url

    return init_chat_model(model_name, **params)
