"""Core data types for the RAG agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set
from pydantic import BaseModel

SourceType = Literal["local", "web"]


@dataclass
class ContextChunk:
    """A chunk of context retrieved from local or web sources."""
    id: str
    content: str
    source_type: SourceType
    source_id: str
    title: Optional[str] = None
    similarity: float = 0.0
    recency: float = 0.5
    reliability: float = 0.5
    metadata: Dict[str, str] = field(default_factory=dict)
    citation: Optional[str] = None


@dataclass
class Answer:
    """Final answer with citations and confidence."""
    text: str
    citations: List[str]
    confidence: float
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class CitationInfo:
    """引用信息，包含编号、标题、来源和页码。"""
    ref: int  # 引用编号 [1], [2], etc.
    title: str  # 章节/表格标题
    source_id: str  # 来源 ID
    source_type: str  # local / sql_database
    doc_type: str  # 文档类型: text / table / sql
    page: str  # 页码信息
    score: float  # 相关性分数
    reliability: float  # 可靠性分数


# ============ 多跳推理辅助类型 ============

class QueryRecord(BaseModel):
    """查询历史记录"""
    hop: int                    # 第几跳
    query: str                  # 查询内容
    intent: str                 # 意图类型 (initial/followup/web)
    result_count: int           # 返回结果数
    new_context_count: int      # 新增上下文数


class MissingInfo(BaseModel):
    """缺失信息描述"""
    type: str                   # 缺失类型 (entity/fact/relation/temporal)
    description: str            # 具体描述
    priority: int               # 优先级 (1-5, 5最高)
    suggested_query: str        # 建议搜索词


class TerminationInfo(BaseModel):
    """终止判断信息"""
    should_terminate: bool      # 是否应该终止
    reason: str                 # 终止原因 (sufficient/max_hops/no_gain/diminishing)
    details: str                # 详细说明


class RetrievalCache(BaseModel):
    """检索缓存（单次检索的中间结果）"""
    sql_results: Optional[str] = None
    vector_results: List[ContextChunk] = []
    bm25_results: List[ContextChunk] = []
    fused_results: List[ContextChunk] = []
    reranked_results: List[ContextChunk] = []
    
    class Config:
        arbitrary_types_allowed = True
