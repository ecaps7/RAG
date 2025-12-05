"""Local retrieval components with hybrid search."""

from .search import (
    # 适配器（兼容 RagAgent）
    LocalRetriever,
    # 主搜索引擎
    HybridSearchEngine,
    get_search_engine,
    hybrid_search,
    # 子组件
    SQLRouter,
    VectorSearcher,
    BM25Searcher,
    # 融合算法
    reciprocal_rank_fusion,
    # 数据类型
    SearchResult,
    SQLResult,
)

__all__ = [
    # 适配器
    "LocalRetriever",
    # 主引擎
    "HybridSearchEngine",
    "get_search_engine",
    "hybrid_search",
    # 子组件
    "SQLRouter",
    "VectorSearcher",
    "BM25Searcher",
    # 融合算法
    "reciprocal_rank_fusion",
    # 数据类型
    "SearchResult",
    "SQLResult",
]
