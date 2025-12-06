"""
搜索器模块

提供不同类型的搜索实现：
- VectorSearcher: 基于 Milvus/Ollama 的向量搜索
- BM25Searcher: 基于 Jieba/BM25 的关键词搜索  
- SQLRouter: 基于 SQLite 的结构化数据查询
"""

from .vector import VectorSearcher
from .keyword import BM25Searcher
from .sql import SQLRouter

__all__ = ["VectorSearcher", "BM25Searcher", "SQLRouter"]
