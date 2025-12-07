"""Core data types for the RAG agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional

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
