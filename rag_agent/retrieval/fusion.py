"""Fusion layer for combining retrieval results from multiple sources."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ..config import TOP_K, get_weights, get_config
from ..core.types import ContextChunk, FusionResult, Intent
from ..utils.text import tokenize_zh
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step

# Import table-aware utilities
try:
    from .local.table_aware import (
        is_table_query,
        score_table_relevance,
        extract_time_from_query,
        is_comparison_query,
    )
    _TABLE_AWARE_AVAILABLE = True
except ImportError:
    _TABLE_AWARE_AVAILABLE = False


def _score_chunk(ch: ContextChunk, w_sim: float, w_rel: float, w_rec: float) -> float:
    """Calculate weighted score for a chunk."""
    return w_sim * ch.similarity + w_rel * ch.reliability + w_rec * ch.recency


def _dedup(chunks: Iterable[ContextChunk]) -> List[ContextChunk]:
    """Remove duplicate chunks based on source and citation."""
    seen: set[str] = set()
    uniq: List[ContextChunk] = []
    for ch in chunks:
        key = (ch.source_type, ch.source_id, ch.citation or ch.title or "")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ch)
    return uniq


def _normalize_list(values: List[float], method: str = "minmax", eps: float = 1e-8) -> List[float]:
    """Normalize a list of values using min-max or z-score normalization."""
    if not values:
        return []
    
    if method == "zscore":
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
        std = (var ** 0.5)
        if std < eps:
            return [0.5 for _ in values]
        out = []
        for v in values:
            z = (v - mean) / (std + eps)
            z = max(-3.0, min(3.0, z))
            out.append((z + 3.0) / 6.0)
        return out
    
    # Default: min-max
    vmin = min(values)
    vmax = max(values)
    rng = vmax - vmin
    if rng < eps:
        return [0.5 for _ in values]
    return [(v - vmin) / (rng + eps) for v in values]


def _chunk_similarity(a: ContextChunk, b: ContextChunk, stopwords: set[str]) -> float:
    """Calculate Jaccard similarity between two chunks."""
    ta = set(tokenize_zh((a.content or "").lower())) - set(stopwords)
    tb = set(tokenize_zh((b.content or "").lower())) - set(stopwords)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / max(len(union), 1)


def _compute_table_boost(
    chunk: ContextChunk, 
    query: str,
    is_table_q: bool,
) -> float:
    """Compute table relevance boost for a chunk.
    
    Args:
        chunk: The context chunk
        query: User query
        is_table_q: Whether query is table-focused
        
    Returns:
        Boost score (0.0 to 0.3)
    """
    if not _TABLE_AWARE_AVAILABLE:
        return 0.0
    
    if not is_table_q:
        return 0.0
    
    return score_table_relevance(chunk, query)


def _has_shareholder_keywords(content: str) -> bool:
    """Check if content contains shareholder-related keywords."""
    keywords = ["股东", "持股", "股份", "比例", "股东情况", "前三名", "前十名"]
    return any(kw in content for kw in keywords)


def _has_definition_keywords(content: str, query: str) -> bool:
    """Check if content contains definition-related keywords matching query context."""
    # Definition indicator patterns
    definition_patterns = ["包括", "包含", "定义", "指的是", "是指", "分为", "由以下", "分别为"]
    has_definition = any(p in content for p in definition_patterns)
    
    if not has_definition:
        return False
    
    # Check if query subject appears in content
    # Extract key terms from query (e.g., "FPA", "融资总量")
    query_terms = []
    if "FPA" in query.upper():
        query_terms.append("FPA")
    if "融资" in query:
        query_terms.append("融资")
    if "客户" in query:
        query_terms.append("客户")
    
    content_upper = content.upper()
    return any(term.upper() in content_upper for term in query_terms) if query_terms else has_definition


def _is_comparison_query(query: str) -> bool:
    """Check if query implies a comparison between entities."""
    if not query:
        return False
    
    # Use table-aware comparison detection if available
    if _TABLE_AWARE_AVAILABLE:
        try:
            return is_comparison_query(query)
        except Exception:
            pass
            
    keywords = ["和", "与", "vs", "对比", "比较", "区别", "差异", "谁", "哪个", "高", "低", "多", "少", "分别"]
    return any(kw in query for kw in keywords)


def _mmr_select(
    candidates: List[ContextChunk],
    scores: Dict[str, float],
    k: int,
    alpha: float,
    stopwords: set[str],
    query: str = ""
) -> List[ContextChunk]:
    """Select top-K chunks using Maximal Marginal Relevance.
    
    score(i) = alpha * relevance(i) - (1-alpha) * max_{j∈S} sim(i, j)
    """
    if not candidates:
        return []
    
    selected: List[ContextChunk] = []
    pool: List[ContextChunk] = list(candidates)

    while pool and len(selected) < k:
        if not selected:
            # First item: pick highest score
            best = max(pool, key=lambda ch: scores.get(ch.id, 0.0))
            selected.append(best)
            pool.remove(best)
            continue
        
        best_item = None
        best_val = float("-inf")
        for ch in pool:
            rel = scores.get(ch.id, 0.0)
            div_penalty = 0.0
            try:
                div_penalty = max(_chunk_similarity(ch, s, stopwords) for s in selected)
            except Exception:
                div_penalty = 0.0
            
            # Reduce diversity penalty for highly relevant chunks (high rel score)
            # or chunks containing critical keywords
            effective_alpha = alpha
            content = ch.content or ""
            if rel > 0.7 or _has_shareholder_keywords(content) or _has_definition_keywords(content, query):
                effective_alpha = min(0.8, alpha + 0.3)  # Boost relevance weight
            
            # For comparison queries, reduce diversity penalty to keep similar entities
            if _is_comparison_query(query):
                effective_alpha = 1.0  # Disable diversity penalty completely

            val = effective_alpha * rel - (1.0 - effective_alpha) * div_penalty
            if val > best_val:
                best_val = val
                best_item = ch
        
        if best_item is None:
            break
        selected.append(best_item)
        pool.remove(best_item)
    
    return selected


class FusionLayer:
    """Fuses retrieval results from multiple sources with intent-based weighting."""
    
    def __init__(self, trace_id: str | None = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)

    @traceable_step("fusion", run_type="chain")
    def aggregate(
        self,
        local: List[ContextChunk],
        web: List[ContextChunk],
        intent: Intent,
        k: int | None = None,
        query: Optional[str] = None,
    ) -> FusionResult:
        """Aggregate and rank chunks from local and web sources.
        
        Args:
            local: Chunks from local retrieval
            web: Chunks from web retrieval
            intent: The classified intent (affects weighting)
            k: Maximum number of results
            query: Original query (for table-aware scoring)
            
        Returns:
            FusionResult with selected chunks and scores
        """
        weights = get_weights(intent)
        w_sim, w_rel, w_rec = weights["w_sim"], weights["w_rel"], weights["w_rec"]
        cfg = get_config()

        merged = _dedup([*local, *web])
        if not merged:
            self.logger.info("Fusion selected 0 chunks (empty inputs)")
            return FusionResult(selected_chunks=[], scores={})

        # Check if query is table-focused
        is_table_q = False
        if query and _TABLE_AWARE_AVAILABLE:
            is_table_q = is_table_query(query)
            if is_table_q:
                self.logger.info("Table-focused query detected, applying table boost")

        # Normalize and score
        scores: Dict[str, float] = {}
        if getattr(cfg, "fusion_use_normalization", True):
            method = str(getattr(cfg, "fusion_norm_method", "minmax")).lower()
            eps = float(getattr(cfg, "fusion_norm_eps", 1e-8))
            
            sim_vals = [ch.similarity for ch in merged]
            rec_vals = [ch.recency for ch in merged]
            rel_vals = [ch.reliability for ch in merged]

            sim_n = _normalize_list(sim_vals, method=method, eps=eps)
            rec_n = _normalize_list(rec_vals, method=method, eps=eps)
            rel_n = _normalize_list(rel_vals, method=method, eps=eps)

            for i, ch in enumerate(merged):
                base_score = w_sim * sim_n[i] + w_rel * rel_n[i] + w_rec * rec_n[i]
                # Apply table boost if applicable
                table_boost = _compute_table_boost(ch, query or "", is_table_q)
                s = base_score + table_boost
                scores[ch.id] = float(s)
        else:
            for ch in merged:
                base_score = _score_chunk(ch, w_sim, w_rel, w_rec)
                table_boost = _compute_table_boost(ch, query or "", is_table_q)
                s = base_score + table_boost
                scores[ch.id] = float(s)

        # Sort by score
        merged.sort(key=lambda c: scores.get(c.id, 0.0), reverse=True)

        top_k = int(k) if isinstance(k, int) and k and k > 0 else int(TOP_K["fusion"])
        use_mmr = bool(getattr(cfg, "fusion_use_mmr", True))
        
        if use_mmr:
            mult = float(getattr(cfg, "fusion_mmr_fetch_multiplier", 2.5))
            alpha = float(getattr(cfg, "fusion_mmr_alpha", 0.35))
            top_n = max(top_k, int(max(1.0, mult) * top_k))
            candidates = merged[:top_n]
            selected = _mmr_select(candidates, scores, top_k, alpha, cfg.zh_stopwords, query or "")
        else:
            selected = merged[:top_k]

        avg_conf = sum(scores.get(ch.id, 0.0) for ch in selected) / max(len(selected), 1)
        table_count = sum(1 for ch in selected if (ch.metadata or {}).get("doctype") == "table")
        self.logger.info(
            "Fusion selected %d chunks (avg=%.2f, mmr=%s, tables=%d)",
            len(selected), avg_conf, str(use_mmr), table_count
        )
        return FusionResult(selected_chunks=selected, scores=scores)
