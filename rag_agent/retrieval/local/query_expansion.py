"""LLM-based query expansion for improved retrieval."""

from __future__ import annotations

import re
from typing import List, Tuple

from ...config import get_config, init_model_with_config
from ...generation.prompts import QUERY_EXPANSION_PROMPT
from ...utils.logging import get_logger


# Cache for expanded queries to avoid repeated LLM calls
_query_expansion_cache: dict[str, List[str]] = {}


# Singleton model instance for query expansion
_query_expansion_model = None


def _get_query_expansion_model():
    """Get or initialize the query expansion LLM model (singleton)."""
    global _query_expansion_model
    if _query_expansion_model is None:
        cfg = get_config()
        # Use the same model as response generation (ChatDeepSeek)
        _query_expansion_model = init_model_with_config(
            cfg.response_model_name,
            temperature=0.3  # Lower temperature for focused query expansion
        )
    return _query_expansion_model


def expand_query_with_llm(
    query: str,
    model: str | None = None,
    num_expansions: int = 4,
) -> List[str]:
    """Use LLM to generate multiple search queries from a user question.
    
    Args:
        query: The original user query
        model: Optional LLM model name (defaults to config, uses ChatDeepSeek)
        num_expansions: Target number of expanded queries
        
    Returns:
        List of expanded queries (including the original)
    """
    logger = get_logger("QueryExpansion")
    
    # Check cache first
    cache_key = f"{query}:{num_expansions}"
    if cache_key in _query_expansion_cache:
        return _query_expansion_cache[cache_key]
    
    # Always include original query
    expanded = [query]
    
    try:
        # Get the LLM model (ChatDeepSeek via LangChain)
        llm = _get_query_expansion_model()
        
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        
        # Use LangChain invoke method
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = llm.invoke(messages)
        
        # Extract text content from response
        response_text = getattr(response, "content", str(response))
        if isinstance(response_text, list):
            response_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response_text
            )
        
        # Parse response - each line is a query
        lines = [line.strip() for line in response_text.strip().split("\n")]
        for line in lines:
            # Clean up the line
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)  # Remove numbering/bullets
            line = line.strip()
            
            if line and line != query and len(line) > 2:
                expanded.append(line)
                if len(expanded) >= num_expansions + 1:  # +1 for original
                    break
        
        logger.info("Expanded query into %d variants", len(expanded))
        
    except Exception as e:
        logger.warning("Query expansion failed: %s, using original query only", e)
    
    # Cache the result
    _query_expansion_cache[cache_key] = expanded
    
    return expanded


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, object]]],
    k: int = 60,
) -> List[Tuple[float, object]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    RRF score = sum(1 / (k + rank_i)) for each list where doc appears
    
    Args:
        ranked_lists: List of ranked document lists, each item is (doc_id, doc_object)
        k: RRF constant (default 60, as in original paper)
        
    Returns:
        List of (rrf_score, doc_object) tuples, sorted by score descending
    """
    doc_scores: dict[str, float] = {}
    doc_objects: dict[str, object] = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, doc_obj) in enumerate(ranked_list, start=1):
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
            doc_objects[doc_id] = doc_obj
    
    # Sort by RRF score descending
    sorted_docs = sorted(
        [(score, doc_objects[doc_id]) for doc_id, score in doc_scores.items()],
        key=lambda x: x[0],
        reverse=True
    )
    
    return sorted_docs


def clear_expansion_cache():
    """Clear the query expansion cache."""
    _query_expansion_cache.clear()
