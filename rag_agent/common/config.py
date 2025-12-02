from __future__ import annotations

from typing import Dict
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from langchain.chat_models import init_chat_model

from .types import Intent


SOURCE_RELIABILITY: Dict[str, float] = {
    "local": 0.9,
    "web": 0.6,
}


TOP_K: Dict[str, int] = {
    "local": 5,
    "web": 5,
    "fusion": 6,
}


# Base weights for fusion layer
DEFAULT_WEIGHTS = {"w_sim": 0.6, "w_rel": 0.3, "w_rec": 0.1}


INTENT_WEIGHTS: Dict[Intent, Dict[str, float]] = {
    Intent.external_context: {"w_sim": 0.55, "w_rel": 0.2, "w_rec": 0.25},
    Intent.forecast: {"w_sim": 0.55, "w_rel": 0.2, "w_rec": 0.25},
    Intent.definition_lookup: {"w_sim": 0.4, "w_rel": 0.5, "w_rec": 0.1},
    Intent.meta_query: {"w_sim": 0.35, "w_rel": 0.55, "w_rec": 0.1},
    Intent.reasoning: {"w_sim": 0.55, "w_rel": 0.25, "w_rec": 0.20},
    Intent.data_lookup: DEFAULT_WEIGHTS,
}


def get_weights(intent: Intent) -> Dict[str, float]:
    return INTENT_WEIGHTS.get(intent, DEFAULT_WEIGHTS)


# Recency decay configuration (used by fusion layer utilities)
RECENCY_HALFLIFE_DAYS = 365  # one year


# ===== 模型与应用配置 =====
load_dotenv()


@dataclass
class AppConfig:
    # ===== 模型与温度 =====
    response_model_name: str = os.getenv("RESPONSE_MODEL", "deepseek:deepseek-chat")
    response_model_temperature: float = float(os.getenv("RESPONSE_TEMPERATURE", "0.7"))

    # ===== API Keys / Base URLs（按提供商可选） =====
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    ark_api_key: str | None = os.getenv("ARK_API_KEY")
    ark_base_url: str | None = os.getenv("ARK_BASE_URL")

    # ===== 检索与向量库配置（与 src/config 对齐的必要子集） =====
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "outputs/vector_store")
    all_chunks_path: str = os.getenv("ALL_CHUNKS_PATH", "outputs/all_chunks.jsonl")
    rebuild_vector_store: bool = os.getenv("REBUILD_VECTOR_STORE", "false").lower() == "true"
    hf_model: str = os.getenv("HF_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "8"))
    use_mmr: bool = os.getenv("USE_MMR", "true").lower() == "true"
    mmr_lambda_mult: float = float(os.getenv("MMR_LAMBDA_MULT", "0.3"))
    mmr_fetch_multiplier: float = float(os.getenv("MMR_FETCH_MULTIPLIER", "3.0"))
    # 交叉编码器重排
    use_cross_encoder: bool = os.getenv("USE_CROSS_ENCODER", "true").lower() == "true"
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
    # BM25 配置与持久化
    bm25_index_path: str = os.getenv("BM25_INDEX_PATH", os.path.join(os.getenv("OUTPUTS_DIR", "outputs"), "bm25_index.json"))
    bm25_k1: float = float(os.getenv("BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("BM25_B", "0.75"))

    # ===== 中文停用词与来源可靠性 =====
    zh_stopwords: set[str] = field(default_factory=lambda: {
        "的", "是", "在", "和", "与", "了", "等", "有", "为", "也", "我", "你", "他", "她",
        "及", "并", "对", "于", "或", "被", "其", "与其", "其中",
    })
    # 与 src/config 的结构保持基本兼容（本地/简单 Web 档位）
    source_reliability: Dict[str, object] = field(default_factory=lambda: {
        "local": SOURCE_RELIABILITY.get("local", 0.9),
        "web": {
            "gov.cn": 0.95,
            "default": SOURCE_RELIABILITY.get("web", 0.6),
        },
    })
    # 每源的 top_k（用于路由层/调用者覆盖）
    top_k_per_source: Dict[str, int] = field(default_factory=lambda: TOP_K.copy())

    # ===== 融合层归一化与多样化（MMR）配置 =====
    fusion_use_normalization: bool = os.getenv("FUSION_USE_NORMALIZATION", "true").lower() == "true"
    fusion_norm_method: str = os.getenv("FUSION_NORM_METHOD", "minmax")  # "minmax" | "zscore"
    fusion_norm_eps: float = float(os.getenv("FUSION_NORM_EPS", "1e-8"))
    fusion_use_mmr: bool = os.getenv("FUSION_USE_MMR", "true").lower() == "true"
    fusion_mmr_alpha: float = float(os.getenv("FUSION_MMR_ALPHA", "0.35"))
    fusion_mmr_fetch_multiplier: float = float(os.getenv("FUSION_MMR_FETCH_MULTIPLIER", "2.5"))


def get_config() -> AppConfig:
    return AppConfig()


def init_model_with_config(model_name: str, temperature: float | None = None):
    """根据环境变量初始化聊天模型。兼容 Doubao/DeepSeek/Google 等提供商。"""
    cfg = get_config()

    api_key = None
    base_url = None

    # 简单提供商路由：包含 doubao 则走 ARK
    if "doubao" in model_name.lower():
        api_key = cfg.ark_api_key
        base_url = cfg.ark_base_url
    else:
        # 默认优先 DeepSeek；若未配置则回退 Google
        api_key = cfg.deepseek_api_key or cfg.google_api_key

    params: Dict[str, object] = {
        "api_key": api_key,
        "temperature": temperature if temperature is not None else cfg.response_model_temperature,
    }
    if base_url:
        params["base_url"] = base_url

    return init_chat_model(model_name, **params)