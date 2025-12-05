"""Retrieval module for local and web search."""

from .base import BaseRetriever
from .fusion import FusionLayer
from .reranker import cross_encoder_rerank

# Local retrieval components - hybrid search engine
from .local import (
    # 适配器（兼容 RagAgent）
    LocalRetriever,
    # 主搜索引擎
    HybridSearchEngine,
    get_search_engine,
    hybrid_search,
    SQLRouter,
    VectorSearcher,
    BM25Searcher,
    reciprocal_rank_fusion,
    SearchResult,
    SQLResult,
)

# Web retrieval components
from .web import WebRetriever

__all__ = [
    # Base
    "BaseRetriever",
    # Fusion
    "FusionLayer",
    # Reranking
    "cross_encoder_rerank",
    # Local retrieval - hybrid search
    "LocalRetriever",
    "HybridSearchEngine",
    "get_search_engine",
    "hybrid_search",
    "SQLRouter",
    "VectorSearcher",
    "BM25Searcher",
    "reciprocal_rank_fusion",
    "SearchResult",
    "SQLResult",
    # Web retrieval
    "WebRetriever",
]
