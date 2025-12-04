"""Cross-encoder reranking with batched inference and Ollama support."""

from __future__ import annotations

from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None  # type: ignore
    _CROSS_ENCODER_AVAILABLE = False

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None  # type: ignore
    _OLLAMA_AVAILABLE = False


# Cache for cross-encoder models
_CROSS_ENCODER_CACHE: dict[str, "CrossEncoder"] = {}

# Default batch size for inference (tune based on GPU memory)
DEFAULT_BATCH_SIZE = 32


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
    
    if not _CROSS_ENCODER_AVAILABLE:
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


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def ollama_rerank(
    query: str,
    pairs: List[Tuple[str, object]],
    model_name: str,
    base_url: str = "http://localhost:11434",
) -> List[Tuple[float, str, object]]:
    """Rerank (doc_text, payload) pairs using Ollama embedding model.
    
    Uses cosine similarity between query and document embeddings for scoring.
    This is the correct way to use embedding models like bge-m3 for reranking.
    
    Args:
        query: The search query
        pairs: List of (doc_text, payload) tuples
        model_name: Name of the Ollama embedding model to use
        base_url: Ollama server base URL
        
    Returns:
        List of (score, doc_text, payload) tuples, sorted by score descending.
    """
    from ..utils.debug import is_debug_enabled
    
    if not _OLLAMA_AVAILABLE or not pairs:
        return []
    
    try:
        client = ollama.Client(host=base_url)
        
        # Get query embedding
        query_response = client.embed(model=model_name, input=query)
        query_embedding = query_response.get("embeddings", [[]])[0]
        
        if not query_embedding:
            if is_debug_enabled():
                print(f"  ⚠️ Failed to get query embedding", flush=True)
            return []
        
        results = []
        
        # Get document embeddings and compute similarity
        # Batch embedding for efficiency
        doc_texts = [doc_text[:2000] for doc_text, _ in pairs]  # Truncate for embedding
        
        try:
            # Try batch embedding first
            doc_response = client.embed(model=model_name, input=doc_texts)
            doc_embeddings = doc_response.get("embeddings", [])
            
            for i, ((doc_text, payload), doc_embedding) in enumerate(zip(pairs, doc_embeddings)):
                if doc_embedding:
                    score = _cosine_similarity(query_embedding, doc_embedding)
                else:
                    score = 0.0
                results.append((score, doc_text, payload))
                
        except Exception:
            # Fallback to individual embedding calls
            for doc_text, payload in pairs:
                try:
                    doc_response = client.embed(model=model_name, input=doc_text[:2000])
                    doc_embedding = doc_response.get("embeddings", [[]])[0]
                    if doc_embedding:
                        score = _cosine_similarity(query_embedding, doc_embedding)
                    else:
                        score = 0.0
                except Exception:
                    score = 0.0
                results.append((score, doc_text, payload))
        
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return results
        
    except Exception as e:
        if is_debug_enabled():
            print(f"  ⚠️ Ollama rerank failed: {e}", flush=True)
        return []


def cross_encoder_rerank(
    query: str,
    pairs: List[Tuple[str, object]],
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Tuple[float, str, object]]:
    """Rerank (doc_text, payload) pairs using a cross-encoder with batched inference.
    
    Args:
        query: The search query
        pairs: List of (doc_text, payload) tuples
        model_name: Name of the cross-encoder model
        batch_size: Number of pairs to process in each batch (default: 32)
        
    Returns:
        List of (score, doc_text, payload) tuples, sorted by score descending.
        Returns empty list if library or model is unavailable.
    """
    model = get_or_create_cross_encoder(model_name)
    if model is None:
        return []
    
    if not pairs:
        return []
    
    try:
        inputs = [(query, text) for (text, _payload) in pairs]
        
        # Batched inference to avoid memory issues with large document sets
        if len(inputs) <= batch_size:
            # Small enough to process in one batch
            scores = list(model.predict(inputs))
        else:
            # Process in batches
            scores: List[float] = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i : i + batch_size]
                batch_scores = model.predict(batch)
                scores.extend(list(batch_scores))
        
        # Combine scores with original data and sort
        ranked = sorted(
            zip(scores, [t for (t, p) in pairs], [p for (t, p) in pairs]),
            key=lambda x: x[0],
            reverse=True
        )
        return ranked
    except Exception:
        return []


def rerank(
    query: str,
    pairs: List[Tuple[str, object]],
    model_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Tuple[float, str, object]]:
    """Unified rerank function that uses configured backend (Ollama or CrossEncoder).
    
    Args:
        query: The search query
        pairs: List of (doc_text, payload) tuples
        model_name: Optional model name override
        batch_size: Batch size for cross-encoder inference
        
    Returns:
        List of (score, doc_text, payload) tuples, sorted by score descending.
    """
    from ..config import get_config
    
    cfg = get_config()
    backend = cfg.reranker_backend
    
    if backend == "ollama" and _OLLAMA_AVAILABLE:
        model = model_name or cfg.ollama_reranker_model
        return ollama_rerank(query, pairs, model, cfg.ollama_base_url)
    else:
        model = model_name or cfg.cross_encoder_model
        return cross_encoder_rerank(query, pairs, model, batch_size)
