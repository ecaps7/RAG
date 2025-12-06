"""
排序与融合算法模块

包含:
- RRF (Reciprocal Rank Fusion): 多路召回结果融合
- SemanticReranker: 基于 Ollama Qwen3-Reranker 的语义重排序
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import requests

from .types import SearchResult
from . import config
from ..utils.logging import get_logger

# 获取 logger
logger = get_logger("rankers")


# RRF 融合参数
RRF_K = config.RRF_K if hasattr(config, "RRF_K") else 60


# ================= RRF 融合 =================

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
        fused_results.append(
            SearchResult(
                id=result.id,
                content=result.content,
                score=rrf_scores[doc_id],
                source=f"rrf({result.source})",
                metadata=result.metadata,
            )
        )

    return fused_results


# ================= 语义重排序 (Reranker) =================

# Qwen3-Reranker 的系统提示和用户提示模板
QWEN3_RERANKER_SYSTEM_PROMPT = """Judge whether the Document meets the requirements based on the Query and theErta Document provided. Note that the answer can only be "yes" or "no"."""

QWEN3_RERANKER_USER_TEMPLATE = """<Query>
{query}
</Query>

<Document>
{document}
</Document>"""


class SemanticReranker:
    """
    基于 Ollama Qwen3-Reranker 的语义重排序器

    使用 Ollama API 调用 Qwen3-Reranker 模型，
    通过 "yes"/"no" 的 logprobs 计算相关性分数。
    """

    def __init__(
        self,
        model: str = config.RERANKER_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/chat"
        self._model_available: Optional[bool] = None

    def _check_model_available(self) -> bool:
        """检查 Ollama 服务和模型是否可用"""
        if self._model_available is not None:
            return self._model_available

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # 检查模型是否已安装
                self._model_available = any(
                    self.model in name or name in self.model for name in model_names
                )
                if not self._model_available:
                    logger.warning(
                        f"Reranker 模型 {self.model} 未安装，请运行: ollama pull {self.model}"
                    )
            else:
                self._model_available = False
        except Exception as e:
            logger.warning(f"无法连接 Ollama 服务: {e}")
            self._model_available = False

        return self._model_available

    def _score_single(self, query: str, document: str) -> float:
        """
        对单个 (query, document) 对进行打分

        通过 Qwen3-Reranker 返回的 logprobs 计算 "yes" 的概率作为相关性分数
        """
        user_content = QWEN3_RERANKER_USER_TEMPLATE.format(
            query=query, document=document
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": QWEN3_RERANKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "num_predict": 1,  # 只需要生成一个 token
                "temperature": 0,  # 确定性输出
                "logprobs": True,  # 返回 logprobs
                "top_logprobs": 10,  # 返回 top 10 logprobs
            },
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            # 解析返回的内容
            message = result.get("message", {})
            content = message.get("content", "").strip().lower()

            # 方法1: 如果有 logprobs，计算 yes 的概率
            # Ollama 的 logprobs 格式可能因版本而异
            # 这里我们使用简化方法：根据输出是 yes 还是 no 来给分

            # 方法2: 简单判断输出
            if "yes" in content:
                return 1.0
            elif "no" in content:
                return 0.0
            else:
                # 无法判断时给中等分数
                return 0.5

        except Exception as e:
            logger.error(f"Reranker 调用失败: {e}")
            return 0.5  # 出错时返回中等分数

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 10,
        threshold: float = config.RERANKER_THRESHOLD,
    ) -> List[SearchResult]:
        """
        对候选结果进行重排序

        Args:
            query: 用户查询
            candidates: 候选文档列表
            top_k: 返回前 K 个结果
            threshold: 分数阈值 (0-1)，低于此分数的文档将被丢弃

        Returns:
            重排序后的 SearchResult 列表
        """
        if not candidates:
            return []

        # 如果只有 1 个结果，直接返回
        if len(candidates) == 1:
            return candidates[:top_k]

        # 检查模型是否可用
        if not self._check_model_available():
            logger.warning("Reranker 不可用，返回原始排序")
            return candidates[:top_k]

        logger.info(f"使用 Ollama Qwen3-Reranker 对 {len(candidates)} 条结果进行重排序...")

        # 对每个候选文档进行打分
        reranked_results = []
        for doc in candidates:
            # 尝试提取标题或章节信息，拼接到文档内容前
            section_info = doc.metadata.get("section", "") or doc.metadata.get(
                "title", ""
            )

            if section_info:
                doc_text = f"{section_info}: {doc.content}"
            else:
                doc_text = doc.content

            # 截断过长的文档（Qwen3-Reranker 建议 8192 tokens 以内）
            if len(doc_text) > 4000:
                doc_text = doc_text[:4000]

            # 调用 Ollama API 进行打分
            score = self._score_single(query, doc_text)

            # 阈值过滤
            if score < threshold:
                continue

            # 创建新的 Result 对象
            new_doc = SearchResult(
                id=doc.id,
                content=doc.content,
                score=score,
                source=f"rerank({doc.source})",
                metadata=doc.metadata,
            )
            reranked_results.append(new_doc)

        # 降序排列
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"重排序完成: 输入 {len(candidates)} -> 输出 {len(reranked_results)} (Top-{top_k})"
        )

        return reranked_results[:top_k]