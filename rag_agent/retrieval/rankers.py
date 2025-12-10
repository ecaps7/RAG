"""
排序与融合算法模块

包含:
- RRF (Reciprocal Rank Fusion): 多路召回结果融合
- SemanticReranker: 基于 Ollama Qwen3-Reranker 的语义重排序
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

# 在导入 transformers 之前禁用警告
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
transformers.logging.set_verbosity_error()

from .types import SearchResult
from rag_agent.config import get_config
from ..utils.logging import get_logger

# 获取 logger
logger = get_logger("rankers")


# RRF 融合参数
config = get_config()
RRF_K = config.rrf_k


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


class SemanticReranker:
    """
    基于 HuggingFace Qwen3-Reranker-0.6B 的语义重排序器

    使用 transformers 加载 Qwen/Qwen3-Reranker-0.6B 模型，
    通过 "yes"/"no" 的 logprobs 计算相关性分数。
    """

    # 类级别缓存，避免重复加载模型
    _model = None
    _tokenizer = None
    _device = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 8192,
    ):
        self.model_name = model_name or config.reranker_model
        self.max_length = max_length
        self._model_available: Optional[bool] = None
        
        # 确定设备 - MPS 存在兼容性问题，强制使用 CPU
        if device is None:
            if torch.cuda.is_available():
                self._device_str = "cuda"
            elif torch.backends.mps.is_available():
                # MPS 在某些情况下会出现张量维度限制问题，使用 CPU 更稳定
                logger.info("检测到 MPS 设备，但由于兼容性问题，Reranker 将使用 CPU")
                self._device_str = "cpu"
            else:
                self._device_str = "cpu"
        else:
            self._device_str = device

    def _load_model(self) -> bool:
        """懒加载模型和分词器（仅从本地加载）"""
        if SemanticReranker._model is not None:
            return True

        try:
            logger.info(f"正在从本地加载 Reranker 模型: {self.model_name}...")
            
            # 临时禁用所有警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                SemanticReranker._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    padding_side='left',
                    trust_remote_code=True,
                    local_files_only=True,  # 仅从本地加载
                )
                
                # 根据设备选择加载方式，使用 dtype 替代已废弃的 torch_dtype
                if self._device_str == "cuda":
                    SemanticReranker._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=True,  # 仅从本地加载
                    ).eval()
                elif self._device_str == "mps":
                    # MPS 模式：使用 float32 避免精度问题
                    SemanticReranker._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        dtype=torch.float32,
                        trust_remote_code=True,
                        local_files_only=True,  # 仅从本地加载
                    ).to("mps").eval()
                else:
                    SemanticReranker._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        dtype=torch.float32,
                        trust_remote_code=True,
                        local_files_only=True,  # 仅从本地加载
                    ).eval()
            
            SemanticReranker._device = self._device_str
            logger.info(f"Reranker 模型加载完成，设备: {self._device_str}")
            return True
            
        except Exception as e:
            logger.error(f"加载 Reranker 模型失败: {e}")
            return False

    def _check_model_available(self) -> bool:
        """检查模型是否可用"""
        if self._model_available is not None:
            return self._model_available
        
        self._model_available = self._load_model()
        return self._model_available

    def _format_input(self, instruction: str, query: str, doc: str) -> str:
        """格式化输入"""
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _score_batch(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        """
        对一批 (query, document) 对进行打分
        
        Args:
            query: 查询文本
            documents: 文档列表
            instruction: 任务指令，默认为通用检索指令
            
        Returns:
            分数列表 (0-1)
        """
        if not documents:
            return []
            
        if instruction is None:
            instruction = "Given a query, retrieve relevant passages that answer the query"
        
        try:
            tokenizer = SemanticReranker._tokenizer
            model = SemanticReranker._model
            
            # 获取 yes/no token ids
            token_true_id = tokenizer.convert_tokens_to_ids("yes")
            token_false_id = tokenizer.convert_tokens_to_ids("no")
            
            # 构建前缀和后缀
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
            
            # 格式化所有输入
            pairs = [self._format_input(instruction, query, doc) for doc in documents]
            
            # Tokenize
            inputs = tokenizer(
                pairs, 
                padding=False, 
                truncation=True,
                return_attention_mask=False, 
                max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens)
            )
            
            # 添加前缀和后缀 tokens
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
            
            # Padding（不传递 max_length 以避免警告）
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
            
            # 移动到设备
            device = SemanticReranker._device
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            
            # 计算分数
            with torch.no_grad():
                batch_scores = model(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
            
            return scores
            
        except Exception as e:
            # 捕获 MPS 或其他设备相关错误，返回默认分数
            logger.error(f"Reranker 打分失败 (可能是设备兼容性问题): {e}")
            logger.warning(f"返回默认分数 0.5 用于 {len(documents)} 个文档")
            return [0.5] * len(documents)

    def _score_single(self, query: str, document: str) -> float:
        """
        对单个 (query, document) 对进行打分
        """
        scores = self._score_batch(query, [document])
        return scores[0] if scores else 0.3

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 10,
        threshold: Optional[float] = None,
        batch_size: int = 8,
    ) -> List[SearchResult]:
        """
        对候选结果进行重排序

        Args:
            query: 用户查询
            candidates: 候选文档列表
            top_k: 返回前 K 个结果
            threshold: 分数阈值 (0-1)，低于此分数的文档将被丢弃
            batch_size: 批处理大小

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

        # 使用配置的阈值如果未提供
        if threshold is None:
            threshold = config.reranker_threshold

        logger.info(f"使用 Qwen3-Reranker-0.6B 对 {len(candidates)} 条结果进行重排序...")

        # 准备文档文本
        doc_texts = []
        for doc in candidates:
            # 尝试提取标题或章节信息，拼接到文档内容前
            section_info = doc.metadata.get("section", "") or doc.metadata.get(
                "title", ""
            )

            if section_info:
                doc_text = f"{section_info}: {doc.content}"
            else:
                doc_text = doc.content

            # 截断过长的文档（更严格的长度限制以避免 MPS 问题）
            # 使用 2000 字符限制（约 500-1000 tokens）
            if len(doc_text) > 2000:
                doc_text = doc_text[:2000]
            
            doc_texts.append(doc_text)

        # 批量打分（使用更小的批处理大小避免内存问题）
        # 在 MPS/CPU 设备上使用更小的批次
        effective_batch_size = min(batch_size, 4) if self._device_str in ["mps", "cpu"] else batch_size
        all_scores = []
        
        for i in range(0, len(doc_texts), effective_batch_size):
            batch_docs = doc_texts[i:i + effective_batch_size]
            try:
                batch_scores = self._score_batch(query, batch_docs)
                all_scores.extend(batch_scores)
            except Exception as e:
                logger.error(f"批次 {i//effective_batch_size + 1} 打分失败: {e}")
                # 降级处理：使用默认分数
                all_scores.extend([0.5] * len(batch_docs))

        # 构建重排序结果
        reranked_results = []
        for doc, score in zip(candidates, all_scores):
            # print(f"[Rerank] Doc ID: {doc.id}, Score: {score:.4f}")

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