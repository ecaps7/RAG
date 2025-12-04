"""Local retriever combining BM25 and vector search."""

from __future__ import annotations

from typing import List, Tuple
from collections import OrderedDict

from ...config import get_config, SOURCE_RELIABILITY
from ...core.types import ContextChunk
from ...utils.logging import get_logger
from ...utils.text import compute_overlap_ratio, expand_date_expressions
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
            from ..reranker import rerank
            from .vector_search import retrieve as vector_retrieve
            from .table_aware import extract_source_filter, is_comparison_query, extract_report_type_filter, filter_by_report_type
            from .query_expansion import expand_query_with_llm, reciprocal_rank_fusion
            
            self._get_store = get_or_create_vector_store
            self._get_bm25 = get_or_create_bm25_index
            self._rerank = rerank
            self._vector_retrieve = vector_retrieve
            self._extract_source_filter = extract_source_filter
            self._is_comparison_query = is_comparison_query
            self._extract_report_type_filter = extract_report_type_filter
            self._filter_by_report_type = filter_by_report_type
            self._expand_query_llm = expand_query_with_llm
            self._rrf_fusion = reciprocal_rank_fusion
            self._available = True
        except Exception as e:
            self.logger.info("Local retriever unavailable: %s", e)
            self._available = False
            self._extract_source_filter = None
            self._is_comparison_query = None
            self._extract_report_type_filter = None
            self._filter_by_report_type = None
            self._expand_query_llm = None
            self._rrf_fusion = None

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

    def _apply_source_filter(self, docs, query: str, boost_factor: float = 2.0):
        """Apply source filter to boost/prioritize documents matching query source.
        
        Args:
            docs: List of documents (with metadata)
            query: The search query
            boost_factor: Multiplier for fetch count when source filter is active
            
        Returns:
            Filtered and reordered documents
        """
        if not self._extract_source_filter:
            return docs
        
        source_filters = self._extract_source_filter(query)
        if not source_filters:
            return docs
        
        # Separate matching and non-matching documents
        matching = []
        non_matching = []
        
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            source = str(meta.get("source", ""))
            if any(f.upper() in source.upper() for f in source_filters):
                matching.append(d)
            else:
                non_matching.append(d)
        
        # Prioritize matching documents
        return matching + non_matching

    @traceable_step("local_retrieval", run_type="retriever")
    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        """Retrieve relevant chunks using LLM query expansion + hybrid BM25/vector search.
        
        Pipeline:
        1. Use LLM to expand query into multiple search queries
        2. For each expanded query, run BM25 + vector search
        3. Fuse results using Reciprocal Rank Fusion (RRF)
        4. Apply source/report filters
        5. Rerank with cross-encoder
        
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

            cfg = get_config()
            
            # === Step 1: LLM Query Expansion ===
            t_expand_start = time.time()
            expanded_queries = [query]  # Always include original
            
            # Use LLM to generate expanded queries if available
            use_llm_expansion = getattr(cfg, "use_llm_query_expansion", True)
            if use_llm_expansion and self._expand_query_llm:
                try:
                    expanded_queries = self._expand_query_llm(query, num_expansions=3)
                    if is_debug_enabled() and len(expanded_queries) > 1:
                        print(f"  🔄 LLM Query expansion ({len(expanded_queries)} queries):", flush=True)
                        for i, eq in enumerate(expanded_queries):
                            print(f"      q{i+1}: {eq}", flush=True)
                except Exception as e:
                    self.logger.warning("LLM query expansion failed: %s", e)
            
            # Also apply rule-based date expansion (e.g., "9月末" -> "9月末 9月30日")
            expanded_queries = [expand_date_expressions(q) for q in expanded_queries]
            t_expand = time.time() - t_expand_start
            
            if is_debug_enabled() and t_expand > 0.1:
                print(f"  ⏱ Query expansion: {t_expand:.2f}s", flush=True)

            # === Step 2: Extract filters (use original query) ===
            source_filters = []
            is_comparison = False
            report_type_patterns = []
            
            if self._extract_source_filter:
                source_filters = self._extract_source_filter(query)
            if self._is_comparison_query:
                is_comparison = self._is_comparison_query(query)
            if self._extract_report_type_filter:
                report_type_patterns = self._extract_report_type_filter(query)
            
            # Determine fetch count per query
            # Increased from 2x to 4x to improve recall for multi-chunk documents
            fetch_k_per_query = max(top_k * 4, 40)
            if source_filters or report_type_patterns:
                if is_debug_enabled():
                    print(f"  🔍 Source filter: {source_filters}", flush=True)
            
            # Determine lambda_mult for MMR
            lambda_mult = 0.99 if is_comparison else None

            # === Step 3: Multi-query retrieval ===
            t1 = time.time()
            all_bm25_results = []  # List of [(doc_id, doc), ...]
            all_vec_results = []
            
            bm25_idx = self._get_bm25()
            
            for eq in expanded_queries:
                # BM25 search
                if bm25_idx is not None:
                    try:
                        bm25_docs = [(self._doc_key(d), d) for d, _s in bm25_idx.search(eq, k=fetch_k_per_query)]
                        all_bm25_results.append(bm25_docs)
                    except Exception:
                        pass
                
                # Vector search
                try:
                    vec_docs, _ = self._vector_retrieve(store, eq, k=fetch_k_per_query, lambda_mult=lambda_mult)
                    vec_results = [(self._doc_key(d), d) for d in vec_docs]
                    all_vec_results.append(vec_results)
                except Exception:
                    pass
            
            t_retrieval = time.time() - t1
            if is_debug_enabled():
                print(f"  ⏱ Store: {t_store:.2f}s | Multi-query retrieval: {t_retrieval:.2f}s", flush=True)

            # === Step 4: RRF Fusion ===
            t2 = time.time()
            all_ranked_lists = all_bm25_results + all_vec_results
            
            if self._rrf_fusion and len(all_ranked_lists) > 0:
                fused_results = self._rrf_fusion(all_ranked_lists, k=60)
                combined = [doc for (_score, doc) in fused_results]
            else:
                # Fallback: simple merge and dedupe
                combined = []
                seen = set()
                for ranked_list in all_ranked_lists:
                    for doc_id, doc in ranked_list:
                        if doc_id not in seen:
                            seen.add(doc_id)
                            combined.append(doc)
            
            t_fusion = time.time() - t2
            if is_debug_enabled() and t_fusion > 0.01:
                print(f"  ⏱ RRF Fusion: {t_fusion:.3f}s ({len(combined)} docs)", flush=True)

            # === Step 5: Apply filters ===
            if source_filters:
                combined = self._apply_source_filter(combined, query)

            if report_type_patterns and self._filter_by_report_type:
                before_count = len(combined)
                combined = self._filter_by_report_type(combined, query)
                after_count = len(combined)
                if is_debug_enabled() and before_count != after_count:
                    print(f"  📑 Report type filter: {report_type_patterns}, {before_count} -> {after_count} docs", flush=True)

            # === Step 6: Cross-encoder reranking ===
            docs_final = combined
            t_rerank = 0.0
            
            try:
                if getattr(cfg, "use_cross_encoder", False) and len(combined) > 1:
                    t3 = time.time()
                    pairs = [(getattr(d, "page_content", ""), d) for d in combined]
                    # Use original query for reranking (most representative of user intent)
                    ranked = self._rerank(query, pairs)
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
                        print(f"  ⏱ Rerank ({cfg.reranker_backend}): {t_rerank:.2f}s", flush=True)
            except Exception:
                pass

            # === Step 7: Truncate and convert ===
            docs_final = docs_final[:top_k]
            chunks = self._convert_docs(query, docs_final)
            self.logger.info("Retrieved %d chunks (LLM-expanded multi-query)", len(chunks))
            
            # Cache results
            self._cache_put(query, top_k, chunks)
            return chunks

        except Exception as e:
            self.logger.info("Local retrieval failed: %s", e)
            return []
    
    def _doc_key(self, doc) -> str:
        """Generate a unique key for a document."""
        meta = getattr(doc, "metadata", {}) or {}
        return f"{meta.get('source','')}-{meta.get('page','')}-{hash(getattr(doc,'page_content',''))}"
