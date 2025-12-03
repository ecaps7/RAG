"""Cross-encoder reranking."""

from __future__ import annotations

from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore


# Cache for cross-encoder models
_CROSS_ENCODER_CACHE: dict[str, "CrossEncoder"] = {}


def get_or_create_cross_encoder(model_name: str) -> "CrossEncoder | None":
    """Get or create a cached cross-encoder model.
    
    This avoids reloading the model on every rerank call (~9s per load).
    
    Args:
        model_name: Name of the cross-encoder model
        
    Returns:
        Cached CrossEncoder instance or None if unavailable
    """
    import time
    from ..utils.debug import is_debug_enabled
    
    if CrossEncoder is None:
        return None
    
    if model_name in _CROSS_ENCODER_CACHE:
        if is_debug_enabled():
            print(f"  🔄 Using cached cross-encoder: {model_name}", flush=True)
        return _CROSS_ENCODER_CACHE[model_name]
    
    try:
        if is_debug_enabled():
            print(f"  ⏳ Loading cross-encoder: {model_name} ...", flush=True)
        start_time = time.time()
        model = CrossEncoder(model_name)
        elapsed = time.time() - start_time
        _CROSS_ENCODER_CACHE[model_name] = model
        if is_debug_enabled():
            print(f"  ✅ Cross-encoder loaded: {model_name} (took {elapsed:.2f}s)", flush=True)
        return model
    except Exception:
        return None


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
    model = get_or_create_cross_encoder(model_name)
    if model is None:
        return []
    
    try:
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
