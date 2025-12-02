"""Utility functions for rag_agent."""

from .text import (
    tokenize_zh,
    compute_overlap_ratio,
    normalize_terms,
    normalize_numbers,
    SYNONYM_MAP,
)
from .logging import get_logger, TraceAdapter, ColorFormatter
from .embed_and_store import (
    load_jsonl,
    to_documents,
    dedup_by_sha1,
    build_embeddings,
    build_in_memory_store,
    build_or_update_faiss,
    create_vector_store,
)

__all__ = [
    # Text processing
    "tokenize_zh",
    "compute_overlap_ratio",
    "normalize_terms",
    "normalize_numbers",
    "SYNONYM_MAP",
    # Logging
    "get_logger",
    "TraceAdapter",
    "ColorFormatter",
    # Embedding & storage
    "load_jsonl",
    "to_documents",
    "dedup_by_sha1",
    "build_embeddings",
    "build_in_memory_store",
    "build_or_update_faiss",
    "create_vector_store",
]