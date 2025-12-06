"""
向量搜索器 - 基于 Milvus 和 Ollama 的语义相似度搜索
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import ollama
from pymilvus import MilvusClient

from .base import BaseSearcher
from ..config import MILVUS_DB_PATH, MILVUS_COLLECTION, OLLAMA_EMBED_MODEL, EMBEDDING_DIM
from ..types import SearchResult
from ...utils.logging import get_logger


class VectorSearcher(BaseSearcher):
    """
    Milvus 向量搜索器
    
    使用 Ollama 生成查询向量，在 Milvus 中执行相似度搜索
    """

    def __init__(
        self,
        db_path: str = MILVUS_DB_PATH,
        collection_name: str = MILVUS_COLLECTION,
    ):
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

            response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
            embedding = response.get("embedding", [])

            if not embedding or len(embedding) != EMBEDDING_DIM:
                self.logger.warning(
                    f"向量维度异常，预期: {EMBEDDING_DIM}, 实际: {len(embedding)}"
                )
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
                output_fields=["text", "subject", "metadata"],
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

                    search_results.append(
                        SearchResult(
                            id=str(hit.get("id", "")),
                            content=hit.get("entity", {}).get("text", ""),
                            score=float(hit.get("distance", 0.0)),
                            source="vector",
                            metadata=metadata,
                        )
                    )

            self.logger.info(f"向量搜索返回 {len(search_results)} 条结果")
            return search_results

        except Exception as e:
            self.logger.error(f"向量搜索失败: {e}")
            return []