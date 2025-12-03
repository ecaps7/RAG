"""Local retriever combining BM25 and vector search."""

from __future__ import annotations

from typing import List, Tuple
from collections import OrderedDict

from ...config import get_config, SOURCE_RELIABILITY
from ...core.types import ContextChunk
from ...utils.logging import get_logger
from ...utils.text import compute_overlap_ratio
from ...utils.tracing import traceable_step


class LocalRetriever:
    """Local retriever that combines BM25 and vector search with optional reranking."""

    def __init__(self, trace_id: str | None = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self._trace_id = trace_id or ""

        # Lazy imports to avoid hard dependencies
        try:
            from .vectorstore import get_or_create_vector_store
            from .bm25 import get_or_create_bm25_index
            from ..reranker import cross_encoder_rerank
            from .vector_search import retrieve as vector_retrieve
            
            self._get_store = get_or_create_vector_store
            self._get_bm25 = get_or_create_bm25_index
            self._cross_rerank = cross_encoder_rerank
            self._vector_retrieve = vector_retrieve
            self._available = True
        except Exception as e:
            self.logger.info("Local retriever unavailable: %s", e)
            self._available = False

        # LRU cache for queries
        try:
            cfg = get_config()
            self._cache_size = int(getattr(cfg, "query_cache_size", 64))
        except Exception:
            self._cache_size = 64
        self._query_cache: OrderedDict[Tuple[str, int], List[ContextChunk]] = OrderedDict()

    def _convert_docs(self, question: str, docs) -> List[ContextChunk]:
        """Convert LangChain documents to ContextChunks."""
        cfg = get_config()
        chunks: List[ContextChunk] = []
        
        for i, d in enumerate(docs):
            meta = dict(getattr(d, "metadata", {}) or {})
            text = getattr(d, "page_content", "")
            sim = float(compute_overlap_ratio(question, text, cfg.zh_stopwords) or 0.0)
            rel = float((cfg.source_reliability or {}).get("local", SOURCE_RELIABILITY["local"]))
            rec = 0.0  # Local documents have no recency signal
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
        """Get cached results."""
        key = (q.strip(), int(k))
        if key in self._query_cache:
            val = self._query_cache.pop(key)
            self._query_cache[key] = val  # Move to end (LRU)
            return val
        return None

    def _cache_put(self, q: str, k: int, chunks: List[ContextChunk]) -> None:
        """Cache results."""
        key = (q.strip(), int(k))
        self._query_cache[key] = chunks
        while len(self._query_cache) > self._cache_size:
            try:
                self._query_cache.popitem(last=False)
            except Exception:
                break

    @traceable_step("local_retrieval", run_type="retriever")
    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        """Retrieve relevant chunks using hybrid BM25 + vector search.
        
        Args:
            query: The search query
            top_k: Maximum number of results
            
        Returns:
            List of ContextChunk objects
        """
        import time
        from ...utils.debug import is_debug_enabled
        
        if not getattr(self, "_available", False):
            self.logger.info("Local retrieval not available; returning empty list.")
            return []

        # Check cache
        cached = self._cache_get(query, top_k)
        if cached is not None:
            self.logger.info("Cache hit: %d chunks", len(cached))
            if is_debug_enabled():
                print(f"  🔄 Query cache hit: {len(cached)} chunks", flush=True)
            return cached

        try:
            t0 = time.time()
            store = self._get_store()
            if not store:
                self.logger.info("Vector store unavailable; returning empty list.")
                return []
            t_store = time.time() - t0

            # 1) BM25 candidates
            t1 = time.time()
            bm25_docs = []
            try:
                bm25_idx = self._get_bm25()
                if bm25_idx is not None:
                    bm25_docs = [d for d, _s in bm25_idx.search(query, k=top_k)]
            except Exception:
                pass
            t_bm25 = time.time() - t1

            # 2) Vector candidates (MMR)
            t2 = time.time()
            vec_docs, _diagnostics = self._vector_retrieve(store, query, k=top_k)
            t_vector = time.time() - t2

            if is_debug_enabled():
                print(f"  ⏱ Store: {t_store:.2f}s | BM25: {t_bm25:.2f}s | Vector: {t_vector:.2f}s", flush=True)

            # 3) Merge and deduplicate
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

            # 4) Cross-encoder reranking (if available)
            cfg = get_config()
            docs_final = combined
            t_rerank = 0.0
            
            try:
                if getattr(cfg, "use_cross_encoder", False) and len(combined) > 1:
                    t3 = time.time()
                    pairs = [(getattr(d, "page_content", ""), d) for d in combined]
                    ranked = self._cross_rerank(query, pairs, getattr(cfg, "cross_encoder_model", ""))
                    t_rerank = time.time() - t3
                    if ranked:
                        docs_final = [payload for (_score, _text, payload) in ranked]
                    else:
                        # Fallback: sort by overlap
                        docs_final = sorted(
                            combined,
                            key=lambda d: compute_overlap_ratio(
                                query,
                                getattr(d, "page_content", ""),
                                cfg.zh_stopwords
                            ),
                            reverse=True,
                        )
                    if is_debug_enabled() and t_rerank > 0.1:
                        print(f"  ⏱ Cross-encoder rerank: {t_rerank:.2f}s", flush=True)
            except Exception:
                pass
            except Exception:
                pass

            # 5) Truncate and convert
            docs_final = docs_final[:top_k]
            chunks = self._convert_docs(query, docs_final)
            self.logger.info("Retrieved %d chunks (BM25+Vector)", len(chunks))
            
            # Cache results
            self._cache_put(query, top_k, chunks)
            return chunks

        except Exception as e:
            self.logger.info("Local retrieval failed: %s", e)
            return []
