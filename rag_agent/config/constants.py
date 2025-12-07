"""Constant values for the RAG agent."""

from __future__ import annotations

from typing import Dict


# Source reliability scores
SOURCE_RELIABILITY: Dict[str, float] = {
    "local": 0.9,
}


# Top-K retrieval settings
TOP_K: Dict[str, int] = {
    "local": 12,   # 增加初始召回数量
    "fusion": 8,   # 融合后保留更多
}
