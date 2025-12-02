"""Cross-encoder reranking."""

from __future__ import annotations

from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore


def cross_encoder_rerank(
    query: str,
    pairs: List[Tuple[str, object]],
    model_name: str
) -> List[Tuple[float, str, object]]:
    """Rerank (doc_text, payload) pairs using a cross-encoder.
    
    Args:
        query: The search query
        pairs: List of (doc_text, payload) tuples
        model_name: Name of the cross-encoder model
        
    Returns:
        List of (score, doc_text, payload) tuples, sorted by score descending.
        Returns empty list if library or model is unavailable.
    """
    if CrossEncoder is None:
        return []
    try:
        model = CrossEncoder(model_name)
        inputs = [(query, text) for (text, _payload) in pairs]
        scores = model.predict(inputs)
        ranked = sorted(
            zip(list(scores), [t for (t, p) in pairs], [p for (t, p) in pairs]),
            key=lambda x: x[0],
            reverse=True
        )
        return ranked
    except Exception:
        return []
