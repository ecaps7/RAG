"""
混合搜索引擎核心模块

包含:
- LocalRetriever: 组合 SQL 路由 + 向量搜索 + BM25 关键词搜索的本地检索器
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.types import ContextChunk
from ..utils.logging import get_logger

from .config import SQL_DB_PATH, MILVUS_DB_PATH, BM25_INDEX_PATH
from .types import SearchResult
from .rankers import reciprocal_rank_fusion, RRF_K, SemanticReranker
from .searchers.vector import VectorSearcher
from .searchers.keyword import BM25Searcher
from .searchers.sql import SQLRouter


class LocalRetriever:
    """
    本地混合检索器

    组合 SQL 路由 + 向量搜索 + BM25 关键词搜索，使用 RRF 融合结果
    可选：使用 Cross-Encoder 进行语义重排序
    """

    def __init__(
        self,
        sql_db_path: str = SQL_DB_PATH,
        milvus_db_path: str = MILVUS_DB_PATH,
        bm25_index_path: str = BM25_INDEX_PATH,
        trace_id: Optional[str] = None,
        use_reranker: bool = True,
    ):
        self.sql_router = SQLRouter(sql_db_path)
        self.vector_searcher = VectorSearcher(milvus_db_path)
        self.bm25_searcher = BM25Searcher(bm25_index_path)
        self.logger = get_logger("LocalRetriever", trace_id)
        self._last_sql_context: Optional[str] = None

        # RRF 权重配置
        self.rrf_weights = {
            "vector": 0.5,
            "bm25": 0.3,
            "sql": 0.2,
        }

        # 可选：语义重排序器
        self.use_reranker = use_reranker
        self.reranker: Optional[SemanticReranker] = None
        if use_reranker:
            self.reranker = SemanticReranker()

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        use_sql: bool = True,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> List[ContextChunk]:
        """
        执行本地混合检索

        Args:
            question: 用户查询
            top_k: 返回结果数量
            use_sql: 是否启用 SQL 路由
            use_vector: 是否启用向量搜索
            use_bm25: 是否启用 BM25 搜索

        Returns:
            ContextChunk 列表
        """
        result_lists: List[List[SearchResult]] = []
        weights: List[float] = []
        sql_context: Optional[str] = None

        # 1. SQL 路由查询（结构化数据）
        if use_sql and self.sql_router.should_route_to_sql(question):
            self.logger.debug("Routing query to SQL...")
            sql_results = self.sql_router.execute_query(question)
            if sql_results:
                sql_context = self.sql_router.format_results_as_context(sql_results)

                # 将 SQL 结果转换为 SearchResult 格式
                sql_search_results = []
                for i, r in enumerate(sql_results):
                    sql_search_results.append(
                        SearchResult(
                            id=f"sql_{i}_{r.source_table_id}",
                            content=f"{r.company_name} {r.report_period} {r.metric_name}: {r.metric_value:,.2f} {r.unit}",
                            score=1.0,  # SQL 精确匹配给高分
                            source="sql",
                            metadata={
                                "stock_code": r.stock_code,
                                "company_name": r.company_name,
                                "report_period": r.report_period,
                                "metric_name": r.metric_name,
                                "source_table_id": r.source_table_id,
                            },
                        )
                    )

                if sql_search_results:
                    result_lists.append(sql_search_results)
                    weights.append(self.rrf_weights["sql"])
                    self.logger.debug(f"SQL search returned {len(sql_search_results)} results")

        # 2. 向量搜索
        if use_vector:
            self.logger.debug("Performing vector search...")
            vector_results = self.vector_searcher.search(question, top_k=top_k * 2)
            if vector_results:
                result_lists.append(vector_results)
                weights.append(self.rrf_weights["vector"])
                self.logger.debug(f"Vector search returned {len(vector_results)} results")

        # 3. BM25 关键词搜索
        if use_bm25:
            self.logger.debug("Performing BM25 search...")
            bm25_results = self.bm25_searcher.search(question, top_k=top_k * 2)
            if bm25_results:
                result_lists.append(bm25_results)
                weights.append(self.rrf_weights["bm25"])
                self.logger.debug(f"BM25 search returned {len(bm25_results)} results")

        # 4. RRF 融合（仅融合非 SQL 结果）
        # SQL 结果单独保存，不参与 RRF 和重排序
        sql_search_results: List[SearchResult] = []
        non_sql_result_lists: List[List[SearchResult]] = []
        non_sql_weights: List[float] = []

        for i, results in enumerate(result_lists):
            if results and results[0].source == "sql":
                sql_search_results = results
            else:
                non_sql_result_lists.append(results)
                non_sql_weights.append(weights[i])

        if non_sql_result_lists:
            self.logger.debug(f"Performing RRF fusion on {len(non_sql_result_lists)} result lists...")
            fused_results = reciprocal_rank_fusion(
                non_sql_result_lists, k=RRF_K, weights=non_sql_weights
            )
            self.logger.debug(f"RRF fusion returned {len(fused_results)} results")
            # 粗排后取更多结果供重排序使用
            coarse_results = fused_results[: top_k * 3] if self.use_reranker else fused_results[:top_k]
        else:
            coarse_results = []

        # 5. 语义重排序（仅对非 SQL 结果）
        if self.use_reranker and self.reranker and coarse_results:
            self.logger.debug(f"Performing semantic reranking on {len(coarse_results)} results...")
            search_results = self.reranker.rerank(question, coarse_results, top_k=top_k)
            self.logger.debug(f"Semantic reranking returned {len(search_results)} results")
        else:
            search_results = coarse_results[:top_k]

        # 6. 转换为 ContextChunk 格式
        chunks = self._to_chunks(search_results)

        # 保存 SQL 上下文，供后续使用
        self._last_sql_context = sql_context

        # 如果有 SQL 结果，添加为第一个 chunk（SQL 结果不参与重排序，直接置顶）
        if sql_context:
            sql_chunk = ContextChunk(
                id="sql_structured_result",
                content=sql_context,
                source_type="local",
                source_id="sql_database",
                title="结构化数据查询结果",
                similarity=1.0,
                recency=1.0,
                reliability=0.95,
                metadata={"doctype": "sql", "retrieval_source": "sql"},
            )
            # SQL 结果放在最前面，不占用 top_k 名额
            chunks = [sql_chunk] + chunks

        self.logger.info(f"LocalRetriever returned {len(chunks)} results")
        return chunks

    def _to_chunks(self, results: List[SearchResult]) -> List[ContextChunk]:
        """将 SearchResult 转换为 ContextChunk 格式"""
        chunks = []
        for r in results:
            # 从 metadata 中提取信息
            metadata = r.metadata or {}
            doc_type = metadata.get("type", "text")

            # 对于表格类型，优先使用 raw_data（HTML）
            content = r.content
            if doc_type == "table" and metadata.get("raw_data"):
                # 保留原始 HTML 以供 LLM 解析
                content = f"[表格数据]\n{metadata.get('raw_data', r.content)}"

            chunk = ContextChunk(
                id=r.id,
                content=content,
                source_type="local",
                source_id=metadata.get("source_id", r.id),
                title=metadata.get("section", ""),
                similarity=r.score,
                recency=0.8,  # 财报数据相对较新
                reliability=0.9,  # 本地数据可靠性高
                metadata={
                    "doctype": doc_type,
                    "page": metadata.get("page", ""),
                    "section": metadata.get("section", ""),
                    "retrieval_source": r.source,
                },
            )
            chunks.append(chunk)

        return chunks

    @property
    def last_sql_context(self) -> Optional[str]:
        """获取上次搜索的 SQL 结构化结果"""
        return self._last_sql_context


# ================= 便捷函数 =================

_retriever_instance: Optional[LocalRetriever] = None


def get_retriever() -> LocalRetriever:
    """获取检索器单例"""
    global _retriever_instance
    if _retriever_instance is None:
        # 默认启用语义重排序
        _retriever_instance = LocalRetriever(use_reranker=True)
    return _retriever_instance


def hybrid_search(
    query: str,
    top_k: int = 10,
) -> List[ContextChunk]:
    """
    执行混合搜索的便捷函数

    Args:
        query: 用户查询
        top_k: 返回结果数量

    Returns:
        ContextChunk 列表
    """
    retriever = get_retriever()
    return retriever.retrieve(query, top_k=top_k)