"""Retrieval module for hybrid search (Vector + BM25 + SQL)."""

from .engine import (
    # 主检索器
    LocalRetriever,
    # 便捷函数
    get_retriever,
    hybrid_search,
)

from .searchers import (
    # 子组件
    SQLRouter,
    VectorSearcher,
    BM25Searcher,
)

from .rankers import (
    # 融合算法
    reciprocal_rank_fusion,
    RRF_K,
    # 重排序
    SemanticReranker,
)

from .types import (
    # 数据类型
    SearchResult,
    SQLResult,
)

__all__ = [
    # 主检索器
    "LocalRetriever",
    # 便捷函数
    "get_retriever",
    "hybrid_search",
    # 子组件
    "SQLRouter",
    "VectorSearcher",
    "BM25Searcher",
    # 融合算法
    "reciprocal_rank_fusion",
    "RRF_K",
    # 重排序
    "SemanticReranker",
    # 数据类型
    "SearchResult",
    "SQLResult",
]
