"""
Local search engine with hybrid retrieval (Vector + BM25 + SQL).

Implements:
1. SQL routing - detect if query needs structured data lookup
2. Vector search via Milvus - semantic similarity
3. BM25 keyword search - keyword matching
4. RRF (Reciprocal Rank Fusion) - combine multi-source results
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jieba
import numpy as np
import ollama
from pymilvus import MilvusClient

from ...config import get_config, TOP_K
from ...core.types import ContextChunk, Intent
from ...utils.logging import get_logger


# ================= 配置 =================

# 数据库路径配置
SQL_DB_PATH = os.getenv("SQL_DB_PATH", "database/financial_rag.db")
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "database/financial_vectors.db")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "database/bm25_index.pkl")

# Embedding 配置
OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b")
EMBEDDING_DIM = 4096

# RRF 融合参数
RRF_K = 60  # RRF 常数，通常取 60


# ================= 数据结构 =================

@dataclass
class SQLResult:
    """SQL 查询结果"""
    metric_name: str
    metric_value: float
    unit: str
    stock_code: str
    company_name: str
    report_period: str
    source_table_id: str


@dataclass
class SearchResult:
    """统一的搜索结果"""
    id: str
    content: str
    score: float
    source: str  # "vector", "bm25", "sql"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ================= SQL 路由器 =================

class SQLRouter:
    """
    SQL 查询路由器
    
    检测查询是否适合 SQL 查询，并生成相应的 SQL 语句
    """
    
    # 财务指标关键词映射（模糊匹配到精确指标名）
    METRIC_PATTERNS: Dict[str, List[str]] = {
        # 收入类
        "营业收入": ["营业收入", "营收", "收入"],
        "净利息收入": ["净利息收入", "利息收入", "净利息"],
        "非利息净收入": ["非利息净收入", "非息收入", "中间业务收入"],
        "手续费及佣金净收入": ["手续费", "佣金收入", "中收"],
        
        # 利润类
        "净利润": ["净利润", "归母净利润", "归属于股东的净利润"],
        "归属于母公司股东的净利润": ["归母净利", "归属母公司", "归母"],
        
        # 每股指标
        "基本每股收益": ["每股收益", "EPS", "基本每股收益"],
        "每股净资产": ["每股净资产", "BPS"],
        
        # 资产负债类
        "总资产": ["总资产", "资产总额", "资产规模"],
        "总负债": ["总负债", "负债总额"],
        "贷款总额": ["贷款总额", "贷款余额", "各项贷款"],
        "存款总额": ["存款总额", "存款余额", "各项存款"],
        
        # 盈利能力
        "净资产收益率": ["ROE", "净资产收益率", "权益回报率"],
        "总资产收益率": ["ROA", "总资产收益率", "资产回报率"],
        "净利差": ["净利差", "NIM", "利差"],
        "净息差": ["净息差", "NIM"],
        
        # 资产质量
        "不良贷款率": ["不良贷款率", "不良率", "NPL"],
        "拨备覆盖率": ["拨备覆盖率", "拨覆率", "拨备"],
        "关注贷款率": ["关注贷款率", "关注类贷款"],
        
        # 资本充足
        "核心一级资本充足率": ["核心一级资本充足率", "CET1", "核心资本"],
        "一级资本充足率": ["一级资本充足率", "T1"],
        "资本充足率": ["资本充足率", "CAR"],
    }
    
    # 数值查询模式
    NUMERIC_PATTERNS = [
        r"是多少",
        r"多少",
        r"达到",
        r"为\s*[\d\.]+",
        r"同比",
        r"环比",
        r"增长",
        r"下降",
        r"变化",
    ]
    
    # 报告期提取模式
    PERIOD_PATTERNS = [
        (r"(\d{4})年第?([一二三四1234])季度?", lambda m: f"{m.group(1)}-Q{_cn_to_num(m.group(2))}"),
        (r"(\d{4})年?[上]?半年度?", lambda m: f"{m.group(1)}-H1"),
        (r"(\d{4})年度?报|(\d{4})年?年度?", lambda m: f"{(m.group(1) or m.group(2))}-FY"),
        (r"(\d{4})[年\-]?Q([1234])", lambda m: f"{m.group(1)}-Q{m.group(2)}"),
        (r"(\d{4})[年\-]?H([12])", lambda m: f"{m.group(1)}-H{m.group(2)}"),
    ]
    
    def __init__(self, db_path: str = SQL_DB_PATH):
        self.db_path = db_path
        self.logger = get_logger("SQLRouter")
        
    def should_route_to_sql(self, query: str) -> bool:
        """
        判断查询是否应该路由到 SQL
        
        条件:
        1. 包含数值查询模式
        2. 包含财务指标关键词
        """
        query = query.lower()
        
        # 检查是否包含数值查询模式
        has_numeric_intent = any(
            re.search(p, query) for p in self.NUMERIC_PATTERNS
        )
        
        # 检查是否包含财务指标
        has_metric = self._extract_metrics(query) != []
        
        return has_numeric_intent and has_metric
    
    def _extract_metrics(self, query: str) -> List[str]:
        """从查询中提取财务指标名称"""
        metrics = []
        query_lower = query.lower()
        
        for metric_name, patterns in self.METRIC_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in query_lower:
                    metrics.append(metric_name)
                    break
        
        return metrics
    
    def _extract_period(self, query: str) -> Optional[str]:
        """从查询中提取报告期"""
        for pattern, formatter in self.PERIOD_PATTERNS:
            match = re.search(pattern, query)
            if match:
                return formatter(match)
        return None
    
    def _extract_company(self, query: str) -> Optional[str]:
        """从查询中提取公司名/股票代码"""
        # 常见银行简称映射（股票代码不带后缀，查询时会模糊匹配）
        company_map = {
            "招商银行": "600036",
            "招行": "600036",
            "中信银行": "601998",
            "中信": "601998",
            "平安银行": "000001",
            "平安": "000001",
            "工商银行": "601398",
            "工行": "601398",
            "建设银行": "601939",
            "建行": "601939",
            "农业银行": "601288",
            "农行": "601288",
            "中国银行": "601988",
            "中行": "601988",
            "交通银行": "601328",
            "交行": "601328",
            "兴业银行": "601166",
            "兴业": "601166",
            "浦发银行": "600000",
            "浦发": "600000",
            "民生银行": "600016",
            "民生": "600016",
            "光大银行": "601818",
            "光大": "601818",
            "华夏银行": "600015",
            "华夏": "600015",
        }
        
        for name, code in company_map.items():
            if name in query:
                return code
        
        # 提取股票代码（6位数字，可能带后缀如 .SH .SZ）
        code_match = re.search(r"(\d{6})(?:\.(?:SH|SZ|sh|sz))?", query)
        if code_match:
            return code_match.group(1)
        
        return None
    
    def execute_query(self, query: str) -> List[SQLResult]:
        """
        执行 SQL 查询（支持 TRIM 和模糊匹配）
        
        Args:
            query: 用户自然语言查询
            
        Returns:
            SQL 查询结果列表
        """
        if not os.path.exists(self.db_path):
            self.logger.warning(f"SQL 数据库不存在: {self.db_path}")
            return []
        
        metrics = self._extract_metrics(query)
        period = self._extract_period(query)
        company = self._extract_company(query)
        
        if not metrics:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            results = []
            
            for metric in metrics:
                # 首先尝试精确匹配（使用 TRIM 去除空格）
                sql = """
                    SELECT metric_name, metric_value, unit, stock_code, company_name, report_period, source_table_id 
                    FROM financial_metrics 
                    WHERE TRIM(metric_name) = TRIM(?)
                """
                params: List[Any] = [metric]
                
                if period:
                    sql += " AND TRIM(report_period) = TRIM(?)"
                    params.append(period)
                
                if company:
                    # 股票代码支持模糊匹配（可能带有前缀或后缀）
                    sql += " AND (TRIM(stock_code) = TRIM(?) OR stock_code LIKE ?)"
                    params.append(company)
                    params.append(f"%{company}%")
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # 如果精确匹配没有结果，尝试模糊匹配
                if not rows:
                    sql_fuzzy = """
                        SELECT metric_name, metric_value, unit, stock_code, company_name, report_period, source_table_id 
                        FROM financial_metrics 
                        WHERE metric_name LIKE ?
                    """
                    params_fuzzy: List[Any] = [f"%{metric}%"]
                    
                    if period:
                        # 报告期也支持模糊匹配（如 2025-Q1 匹配 2025Q1 或 2025-Q1）
                        period_pattern = period.replace("-", "").replace("_", "")
                        sql_fuzzy += " AND (REPLACE(REPLACE(report_period, '-', ''), '_', '') LIKE ? OR report_period LIKE ?)"
                        params_fuzzy.append(f"%{period_pattern}%")
                        params_fuzzy.append(f"%{period}%")
                    
                    if company:
                        sql_fuzzy += " AND (stock_code LIKE ? OR company_name LIKE ?)"
                        params_fuzzy.append(f"%{company}%")
                        params_fuzzy.append(f"%{company}%")
                    
                    cursor.execute(sql_fuzzy, params_fuzzy)
                    rows = cursor.fetchall()
                    
                    if rows:
                        self.logger.info(f"模糊匹配找到 {len(rows)} 条结果 (指标: {metric})")
                
                for row in rows:
                    results.append(SQLResult(
                        metric_name=row[0].strip() if row[0] else row[0],
                        metric_value=row[1],
                        unit=row[2].strip() if row[2] else row[2],
                        stock_code=row[3].strip() if row[3] else row[3],
                        company_name=row[4].strip() if row[4] else row[4],
                        report_period=row[5].strip() if row[5] else row[5],
                        source_table_id=row[6],
                    ))
            
            # 去重（同一指标可能被多次匹配）
            seen = set()
            unique_results = []
            for r in results:
                key = (r.stock_code, r.report_period, r.metric_name, r.metric_value)
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            conn.close()
            self.logger.info(f"SQL 查询返回 {len(unique_results)} 条结果（去重后）")
            return unique_results
            
        except Exception as e:
            self.logger.error(f"SQL 查询失败: {e}")
            return []
    
    def format_results_as_context(self, results: List[SQLResult]) -> str:
        """将 SQL 结果格式化为上下文文本"""
        if not results:
            return ""
        
        lines = ["【结构化数据查询结果】"]
        for r in results:
            lines.append(
                f"- {r.company_name}({r.stock_code}) {r.report_period} {r.metric_name}: "
                f"{r.metric_value:,.2f} {r.unit}"
            )
        
        return "\n".join(lines)


def _cn_to_num(cn: str) -> str:
    """中文数字转阿拉伯数字"""
    mapping = {"一": "1", "二": "2", "三": "3", "四": "4", "1": "1", "2": "2", "3": "3", "4": "4"}
    return mapping.get(cn, cn)


# ================= 向量搜索 =================

class VectorSearcher:
    """
    Milvus 向量搜索器
    """
    
    def __init__(self, db_path: str = MILVUS_DB_PATH, collection_name: str = "financial_chunks"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client: Optional[MilvusClient] = None
        self.logger = get_logger("VectorSearcher")
        
    def _ensure_client(self) -> bool:
        """确保 Milvus 客户端已初始化"""
        if self.client is not None:
            return True
        
        if not os.path.exists(self.db_path):
            self.logger.warning(f"Milvus 数据库不存在: {self.db_path}")
            return False
        
        try:
            self.client = MilvusClient(uri=self.db_path)
            if not self.client.has_collection(self.collection_name):
                self.logger.warning(f"集合不存在: {self.collection_name}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Milvus 初始化失败: {e}")
            return False
    
    def _get_embedding(self, text: str) -> List[float]:
        """调用 Ollama 获取向量"""
        try:
            text = text.replace("\n", " ").strip()
            if not text:
                return [0.0] * EMBEDDING_DIM
            
            response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
            embedding = response.get('embedding', [])
            
            if not embedding or len(embedding) != EMBEDDING_DIM:
                self.logger.warning(f"向量维度异常，返回零向量")
                return [0.0] * EMBEDDING_DIM
            
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding 调用失败: {e}")
            return [0.0] * EMBEDDING_DIM
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        向量相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self._ensure_client():
            return []
        
        # 获取查询向量
        query_vec = self._get_embedding(query)
        
        try:
            # 执行搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vec],
                limit=top_k,
                output_fields=["text", "subject", "metadata"]
            )
            
            search_results = []
            for hits in results:
                for hit in hits:
                    # 解析 metadata
                    metadata_str = hit.get("entity", {}).get("metadata", "{}")
                    try:
                        metadata = json.loads(metadata_str)
                    except Exception:
                        metadata = {}
                    
                    search_results.append(SearchResult(
                        id=str(hit.get("id", "")),
                        content=hit.get("entity", {}).get("text", ""),
                        score=float(hit.get("distance", 0.0)),
                        source="vector",
                        metadata=metadata,
                    ))
            
            self.logger.info(f"向量搜索返回 {len(search_results)} 条结果")
            return search_results
            
        except Exception as e:
            self.logger.error(f"向量搜索失败: {e}")
            return []


# ================= BM25 关键词搜索 =================

class BM25Searcher:
    """
    BM25 关键词搜索器
    """
    
    def __init__(self, index_path: str = BM25_INDEX_PATH):
        self.index_path = index_path
        self.bm25 = None
        self.doc_map: List[Dict] = []
        self.logger = get_logger("BM25Searcher")
        self._loaded = False
        
    def _ensure_loaded(self) -> bool:
        """确保 BM25 索引已加载"""
        if self._loaded:
            return True
        
        if not os.path.exists(self.index_path):
            self.logger.warning(f"BM25 索引不存在: {self.index_path}")
            return False
        
        try:
            with open(self.index_path, 'rb') as f:
                self.bm25, self.doc_map = pickle.load(f)
            self._loaded = True
            self.logger.info(f"BM25 索引加载完成，共 {len(self.doc_map)} 条文档")
            return True
        except Exception as e:
            self.logger.error(f"BM25 索引加载失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        BM25 关键词搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self._ensure_loaded():
            return []
        
        # 分词
        tokens = list(jieba.cut_for_search(query))
        
        try:
            # 计算 BM25 分数
            scores = self.bm25.get_scores(tokens)
            
            # 获取 top-k 索引
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            search_results = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                
                doc = self.doc_map[idx]
                
                # 解析 metadata
                metadata_str = doc.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                except Exception:
                    metadata = {}
                
                search_results.append(SearchResult(
                    id=metadata.get("source_id", str(idx)),
                    content=doc.get("text", ""),
                    score=float(scores[idx]),
                    source="bm25",
                    metadata=metadata,
                ))
            
            self.logger.info(f"BM25 搜索返回 {len(search_results)} 条结果")
            return search_results
            
        except Exception as e:
            self.logger.error(f"BM25 搜索失败: {e}")
            return []


# ================= RRF 融合算法 =================

def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = RRF_K,
    weights: Optional[List[float]] = None,
) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) 算法
    
    融合多路召回结果，公式:
    RRF_score(d) = Σ (w_i / (k + rank_i(d)))
    
    Args:
        result_lists: 多个搜索结果列表
        k: RRF 常数，通常取 60
        weights: 各路召回的权重，默认均等
        
    Returns:
        融合后的排序结果
    """
    if not result_lists:
        return []
    
    # 默认权重
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    # 归一化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # 计算 RRF 分数
    rrf_scores: Dict[str, float] = {}
    doc_cache: Dict[str, SearchResult] = {}
    
    for i, results in enumerate(result_lists):
        for rank, result in enumerate(results, start=1):
            doc_id = result.id
            rrf_score = weights[i] / (k + rank)
            
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += rrf_score
            else:
                rrf_scores[doc_id] = rrf_score
                doc_cache[doc_id] = result
    
    # 按 RRF 分数排序
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # 构建结果
    fused_results = []
    for doc_id in sorted_ids:
        result = doc_cache[doc_id]
        # 更新分数为 RRF 分数
        fused_results.append(SearchResult(
            id=result.id,
            content=result.content,
            score=rrf_scores[doc_id],
            source=f"rrf({result.source})",
            metadata=result.metadata,
        ))
    
    return fused_results


# ================= 联合搜索引擎 =================

class HybridSearchEngine:
    """
    混合搜索引擎
    
    组合 SQL 路由 + 向量搜索 + BM25 关键词搜索，使用 RRF 融合结果
    """
    
    def __init__(
        self,
        sql_db_path: str = SQL_DB_PATH,
        milvus_db_path: str = MILVUS_DB_PATH,
        bm25_index_path: str = BM25_INDEX_PATH,
        trace_id: Optional[str] = None,
    ):
        self.sql_router = SQLRouter(sql_db_path)
        self.vector_searcher = VectorSearcher(milvus_db_path)
        self.bm25_searcher = BM25Searcher(bm25_index_path)
        self.logger = get_logger("HybridSearchEngine", trace_id)
        
        # RRF 权重配置
        self.rrf_weights = {
            "vector": 0.5,
            "bm25": 0.3,
            "sql": 0.2,
        }
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_sql: bool = True,
        use_vector: bool = True,
        use_bm25: bool = True,
        intent: Optional[Intent] = None,
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """
        执行混合搜索
        
        Args:
            query: 用户查询
            top_k: 最终返回结果数量
            use_sql: 是否启用 SQL 路由
            use_vector: 是否启用向量搜索
            use_bm25: 是否启用 BM25 搜索
            intent: 用户意图（可用于调整搜索策略）
            
        Returns:
            (搜索结果列表, SQL 结构化结果文本)
        """
        result_lists: List[List[SearchResult]] = []
        weights: List[float] = []
        sql_context: Optional[str] = None
        
        # 根据意图调整权重
        if intent == Intent.data_lookup:
            self.rrf_weights = {"vector": 0.4, "bm25": 0.3, "sql": 0.3}
        elif intent == Intent.definition_lookup:
            self.rrf_weights = {"vector": 0.5, "bm25": 0.4, "sql": 0.1}
        elif intent == Intent.reasoning:
            self.rrf_weights = {"vector": 0.6, "bm25": 0.3, "sql": 0.1}
        
        # 1. SQL 路由查询（结构化数据）
        if use_sql and self.sql_router.should_route_to_sql(query):
            self.logger.info("查询路由到 SQL...")
            sql_results = self.sql_router.execute_query(query)
            if sql_results:
                sql_context = self.sql_router.format_results_as_context(sql_results)
                
                # 将 SQL 结果转换为 SearchResult 格式
                sql_search_results = []
                for i, r in enumerate(sql_results):
                    sql_search_results.append(SearchResult(
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
                    ))
                
                if sql_search_results:
                    result_lists.append(sql_search_results)
                    weights.append(self.rrf_weights["sql"])
        
        # 2. 向量搜索
        if use_vector:
            self.logger.info("执行向量搜索...")
            vector_results = self.vector_searcher.search(query, top_k=top_k * 2)
            if vector_results:
                result_lists.append(vector_results)
                weights.append(self.rrf_weights["vector"])
        
        # 3. BM25 关键词搜索
        if use_bm25:
            self.logger.info("执行 BM25 搜索...")
            bm25_results = self.bm25_searcher.search(query, top_k=top_k * 2)
            if bm25_results:
                result_lists.append(bm25_results)
                weights.append(self.rrf_weights["bm25"])
        
        # 4. RRF 融合
        if result_lists:
            self.logger.info(f"RRF 融合 {len(result_lists)} 路召回结果...")
            fused_results = reciprocal_rank_fusion(result_lists, k=RRF_K, weights=weights)
            final_results = fused_results[:top_k]
        else:
            final_results = []
        
        self.logger.info(f"混合搜索返回 {len(final_results)} 条结果")
        return final_results, sql_context
    
    def search_to_chunks(
        self,
        query: str,
        top_k: int = 10,
        intent: Optional[Intent] = None,
    ) -> Tuple[List[ContextChunk], Optional[str]]:
        """
        执行混合搜索并转换为 ContextChunk 格式
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            intent: 用户意图
            
        Returns:
            (ContextChunk 列表, SQL 结构化结果文本)
        """
        results, sql_context = self.search(query, top_k=top_k, intent=intent)
        
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
        
        return chunks, sql_context


# ================= LocalRetriever 适配器 =================

class LocalRetriever:
    """
    本地检索器适配器
    
    兼容现有 RagAgent 接口，内部使用 HybridSearchEngine
    """
    
    def __init__(
        self,
        sql_db_path: str = SQL_DB_PATH,
        milvus_db_path: str = MILVUS_DB_PATH,
        bm25_index_path: str = BM25_INDEX_PATH,
        trace_id: Optional[str] = None,
    ):
        self.engine = HybridSearchEngine(
            sql_db_path=sql_db_path,
            milvus_db_path=milvus_db_path,
            bm25_index_path=bm25_index_path,
            trace_id=trace_id,
        )
        self.logger = get_logger("LocalRetriever", trace_id)
        self._last_sql_context: Optional[str] = None
    
    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        intent: Optional[Intent] = None,
    ) -> List[ContextChunk]:
        """
        执行本地检索
        
        Args:
            question: 用户查询
            top_k: 返回结果数量
            intent: 用户意图（可选）
            
        Returns:
            ContextChunk 列表
        """
        chunks, sql_context = self.engine.search_to_chunks(
            query=question,
            top_k=top_k,
            intent=intent,
        )
        
        # 保存 SQL 上下文，供后续使用
        self._last_sql_context = sql_context
        
        # 如果有 SQL 结果，添加为第一个 chunk
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
            # SQL 结果放在最前面
            chunks = [sql_chunk] + chunks[:top_k - 1]
        
        self.logger.info(f"LocalRetriever 返回 {len(chunks)} 条结果")
        return chunks
    
    @property
    def last_sql_context(self) -> Optional[str]:
        """获取上次搜索的 SQL 结构化结果"""
        return self._last_sql_context


# ================= 便捷函数 =================

_engine_instance: Optional[HybridSearchEngine] = None


def get_search_engine() -> HybridSearchEngine:
    """获取搜索引擎单例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HybridSearchEngine()
    return _engine_instance


def hybrid_search(
    query: str,
    top_k: int = 10,
    intent: Optional[Intent] = None,
) -> Tuple[List[ContextChunk], Optional[str]]:
    """
    执行混合搜索的便捷函数
    
    Args:
        query: 用户查询
        top_k: 返回结果数量
        intent: 用户意图
        
    Returns:
        (ContextChunk 列表, SQL 结构化结果文本)
    """
    engine = get_search_engine()
    return engine.search_to_chunks(query, top_k=top_k, intent=intent)


# ================= CLI 测试 =================

def main():
    """命令行测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="混合搜索引擎测试")
    parser.add_argument("query", type=str, help="搜索查询")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--no-sql", action="store_true", help="禁用 SQL 路由")
    parser.add_argument("--no-vector", action="store_true", help="禁用向量搜索")
    parser.add_argument("--no-bm25", action="store_true", help="禁用 BM25 搜索")
    
    args = parser.parse_args()
    
    engine = HybridSearchEngine()
    
    results, sql_context = engine.search(
        args.query,
        top_k=args.top_k,
        use_sql=not args.no_sql,
        use_vector=not args.no_vector,
        use_bm25=not args.no_bm25,
    )
    
    print("\n" + "=" * 60)
    print(f"查询: {args.query}")
    print("=" * 60)
    
    if sql_context:
        print(f"\n{sql_context}\n")
    
    print(f"\n共 {len(results)} 条结果:\n")
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r.score:.4f} | Source: {r.source}")
        print(f"    ID: {r.id}")
        print(f"    Content: {r.content[:200]}...")
        print(f"    Metadata: {r.metadata}")
        print()


if __name__ == "__main__":
    main()
