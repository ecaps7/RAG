"""Vector search operations with MMR support."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from ...config import get_config
from ...utils.text import compute_overlap_ratio


def _similarity_search_with_score_safe(
    store: Any,
    query: str,
    k: int
) -> Tuple[List[Document], List[float]]:
    """Safe similarity search with score fallback."""
    try:
        pairs = store.similarity_search_with_score(query, k=k)
        docs = [doc for doc, _ in pairs]
        scores = [score for _, score in pairs]
        return docs, scores
    except Exception:
        docs = store.similarity_search(query, k=k)
        return docs, [0.0] * len(docs)


def _mmr_search_safe(
    store: Any,
    query: str,
    k: int,
    fetch_k: int,
    lambda_mult: float
) -> List[Document]:
    """Safe MMR search with empty fallback."""
    try:
        return store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
    except Exception:
        return []


def retrieve(
    vector_store: Any,
    query: str,
    k: Optional[int] = None,
    lambda_mult: Optional[float] = None,
) -> Tuple[List[Document], dict | None]:
    """Execute vector retrieval with MMR or similarity search.
    
    Args:
        vector_store: The vector store to search
        query: The search query
        k: Maximum number of results
        lambda_mult: Optional override for MMR lambda multiplier (0-1).
                     Higher values (e.g. 0.8) favor relevance over diversity.
        
    Returns:
        Tuple of (documents, diagnostics)
    """
    cfg = get_config()
    kk = int(k) if isinstance(k, int) and k and k > 0 else int(cfg.top_k_retrieval)

    if getattr(cfg, "use_mmr", True):
        fetch_k = max(kk, int(max(1.0, float(getattr(cfg, "mmr_fetch_multiplier", 3.0))) * kk))
        
        # Use provided lambda_mult or fallback to config
        if lambda_mult is not None:
            lm = float(lambda_mult)
        else:
            lm = float(getattr(cfg, "mmr_lambda_mult", 0.3))
            
        candidate_docs, candidate_scores = _similarity_search_with_score_safe(vector_store, query, fetch_k)
        docs = _mmr_search_safe(vector_store, query, kk, fetch_k, lm) or candidate_docs[:kk]
        scores = candidate_scores
    else:
        docs, scores = _similarity_search_with_score_safe(vector_store, query, kk)

    top_score = max(scores) if scores else 0.0
    overlaps = [compute_overlap_ratio(query, d.page_content, cfg.zh_stopwords) for d in docs]
    avg_overlap = (sum(overlaps) / len(overlaps)) if overlaps else 0.0

    diagnostics = {"best_score": top_score, "overlap_ratio": avg_overlap}
    return docs, diagnostics
