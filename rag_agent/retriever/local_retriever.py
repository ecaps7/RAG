from __future__ import annotations

from typing import List, Tuple
from collections import OrderedDict

from ..common.config import SOURCE_RELIABILITY
from ..common.logging import get_logger
from ..common.types import ContextChunk


class LocalRetriever:
    """
    本地向量检索，完全使用 rag_agent 内部实现。
    """

    def __init__(self, trace_id: str | None = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self._trace_id = trace_id or ""

        # Lazy imports to avoid hard dependency during package import
        try:
            from ..vectorstore import get_or_create_vector_store  # type: ignore
            from .core import retrieve as rag_retrieve  # type: ignore
            from ..common.utils import compute_overlap_ratio  # type: ignore
            from ..common.config import get_config  # type: ignore
            from .bm25 import get_or_create_bm25_index  # type: ignore
            from .rerankers import cross_encoder_rerank  # type: ignore
            self._get_store = get_or_create_vector_store
            self._retrieve = rag_retrieve
            self._compute_overlap = compute_overlap_ratio
            self._get_cfg = get_config
            self._get_bm25 = get_or_create_bm25_index
            self._cross_rerank = cross_encoder_rerank
            self._available = True
        except Exception as e:
            self.logger.info("Local retriever unavailable, will use empty results. (%s)", e)
            self._available = False

        # 简单查询缓存（LRU）
        try:
            cfg = self._get_cfg()
            self._cache_size = int(getattr(cfg, "query_cache_size", 64))
        except Exception:
            self._cache_size = 64
        self._query_cache: OrderedDict[Tuple[str, int], List[ContextChunk]] = OrderedDict()

    def _convert_docs(self, question: str, docs) -> List[ContextChunk]:
        cfg = self._get_cfg()
        chunks: List[ContextChunk] = []
        for i, d in enumerate(docs):
            meta = dict(getattr(d, "metadata", {}) or {})
            text = getattr(d, "page_content", "")
            sim = float(self._compute_overlap(question, text, cfg.zh_stopwords) or 0.0)
            rel = float((cfg.source_reliability or {}).get("local", SOURCE_RELIABILITY["local"]))
            rec = 0.0  # local documents generally have no recency signal
            source = str(meta.get("source", "local"))
            page = meta.get("page", i)
            citation = f"《{source}》第 {page} 页" if source else f"local#{page}"
            chunks.append(
                ContextChunk(
                    id=f"local-{i}",
                    content=text,
                    source_type="local",
                    source_id=source,
                    title=str(meta.get("title", "")) or None,
                    similarity=sim,
                    reliability=rel,
                    recency=rec,
                    metadata={k: str(v) for k, v in meta.items()},
                    citation=citation,
                )
            )
        return chunks

    def _cache_get(self, q: str, k: int) -> List[ContextChunk] | None:
        key = (q.strip(), int(k))
        if key in self._query_cache:
            val = self._query_cache.pop(key)
            # 触发 LRU 更新顺序
            self._query_cache[key] = val
            return val
        return None

    def _cache_put(self, q: str, k: int, chunks: List[ContextChunk]) -> None:
        key = (q.strip(), int(k))
        self._query_cache[key] = chunks
        # 控制大小
        while len(self._query_cache) > self._cache_size:
            try:
                self._query_cache.popitem(last=False)
            except Exception:
                break

    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        if not getattr(self, "_available", False):
            self.logger.info("Local vector retrieval not available; returning empty list.")
            return []
        # 命中缓存直接返回
        cached = self._cache_get(query, top_k)
        if cached is not None:
            self.logger.info("Local cache hit: %d chunks", len(cached))
            return cached
        try:
            store = self._get_store()
            if not store:
                self.logger.info("Vector store unavailable; returning empty list.")
                return []
            # 1) BM25 候选（优先补充基于词匹配的召回）
            bm25_idx = None
            try:
                bm25_idx = self._get_bm25()
            except Exception:
                bm25_idx = None

            bm25_docs = []
            if bm25_idx is not None:
                try:
                    bm25_docs = [d for d, _s in bm25_idx.search(query, k=top_k)]
                except Exception:
                    bm25_docs = []

            # 2) 向量候选（MMR）
            vec_docs, _diagnostics = self._retrieve(store, query, k=top_k)

            # 3) 合并去重
            combined = []
            seen_keys = set()
            def _key_of(doc) -> str:
                meta = getattr(doc, "metadata", {}) or {}
                return f"{meta.get('source','')}-{meta.get('page','')}-{hash(getattr(doc,'page_content',''))}"

            for d in bm25_docs + vec_docs:
                k = _key_of(d)
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                combined.append(d)

            # 4) 交叉编码器重排（有则用，无则回退简单重叠排序）
            cfg = self._get_cfg()
            docs_final = combined
            try:
                if getattr(cfg, "use_cross_encoder", False) and len(combined) > 1:
                    pairs = [(getattr(d, "page_content", ""), d) for d in combined]
                    ranked = self._cross_rerank(query, pairs, getattr(cfg, "cross_encoder_model", ""))
                    if ranked:
                        docs_final = [payload for (_score, _text, payload) in ranked]
                    else:
                        # 回退：按中文词重叠率排序
                        docs_final = sorted(
                            combined,
                            key=lambda d: self._compute_overlap(query, getattr(d, "page_content", ""), cfg.zh_stopwords),
                            reverse=True,
                        )
            except Exception:
                pass

            # 5) 截断到 top_k 并转换为 ContextChunk
            docs_final = docs_final[:top_k]
            chunks = self._convert_docs(query, docs_final)
            self.logger.info("Local retrieved %d chunks (BM25+Vector)", len(chunks))
            # 写入缓存
            self._cache_put(query, top_k, chunks)
            return chunks
        except Exception as e:
            self.logger.info("Local vector retrieval failed: %s", e)
            return []