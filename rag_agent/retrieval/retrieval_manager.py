"""检索管理器 - 封装所有检索相关功能"""

from typing import List, Optional
from rag_agent.config import get_config
from rag_agent.core.types import ContextChunk
from rag_agent.retrieval.searchers.sql import SQLRouter
from rag_agent.retrieval.searchers.vector import VectorSearcher
from rag_agent.retrieval.searchers.keyword import BM25Searcher
from rag_agent.retrieval.rankers import reciprocal_rank_fusion
from rag_agent.retrieval.types import SearchResult
from rag_agent.retrieval.engine import LocalRetriever


class RetrievalManager:
    """检索管理器 - 封装所有检索相关功能"""
    
    def __init__(self):
        self.sql_router = SQLRouter()
        self.vector_searcher = VectorSearcher()
        self.bm25_searcher = BM25Searcher()
        self.local_retriever = LocalRetriever()
        # 从LocalRetriever中获取reranker
        self.reranker = self.local_retriever.reranker
    
    def should_route_to_sql(self, query: str) -> bool:
        """判断是否需要SQL查询"""
        return self.sql_router.should_route_to_sql(query)
    
    def execute_sql_query(self, query: str) -> Optional[str]:
        """执行SQL查询并返回结果上下文"""
        sql_results = self.sql_router.execute_query(query)
        if sql_results:
            return self.sql_router.format_results_as_context(sql_results)
        return None
    
    def vector_retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """执行向量检索"""
        return self.vector_searcher.search(query, top_k)
    
    def bm25_retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """执行BM25检索"""
        return self.bm25_searcher.search(query, top_k)
    
    def rrf_fusion(self, result_lists: List[List[SearchResult]]) -> List[SearchResult]:
        """执行RRF融合"""
        return reciprocal_rank_fusion(result_lists)
    
    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 8) -> List[SearchResult]:
        """执行语义重排序"""
        if self.reranker:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]
    
    def hybrid_retrieve(self, query: str, top_k: int = 8) -> List[ContextChunk]:
        """执行混合检索"""
        return self.local_retriever.retrieve(query, top_k)
