"""
BM25 关键词搜索器 - 基于 Jieba 分词和 BM25 算法
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List

import jieba
import numpy as np

from .base import BaseSearcher
from rag_agent.config import get_config
from ..types import SearchResult
from ...utils.logging import get_logger


class BM25Searcher(BaseSearcher):
    """
    BM25 关键词搜索器
    
    使用预建的 BM25 索引进行关键词匹配搜索
    """

    def __init__(self, index_path: Optional[str] = None):
        config = get_config()
        self.index_path = index_path or config.bm25_index_path
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
            with open(self.index_path, "rb") as f:
                self.bm25, self.doc_map = pickle.load(f)
            self._loaded = True
            self.logger.debug(f"BM25 索引加载完成，共 {len(self.doc_map)} 条文档")
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

                search_results.append(
                    SearchResult(
                        id=metadata.get("source_id", str(idx)),
                        content=doc.get("text", ""),
                        score=float(scores[idx]),
                        source="bm25",
                        metadata=metadata,
                    )
                )

            self.logger.debug(f"BM25 搜索返回 {len(search_results)} 条结果")
            return search_results

        except Exception as e:
            self.logger.error(f"BM25 搜索失败: {e}")
            return []