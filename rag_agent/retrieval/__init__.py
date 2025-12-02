"""Retrieval module for local and web search."""

from .base import BaseRetriever
from .fusion import FusionLayer
from .reranker import cross_encoder_rerank

# Local retrieval components
from .local import LocalRetriever, BM25Index, get_or_create_bm25_index
from .local.vectorstore import get_or_create_vector_store

# Web retrieval components
from .web import WebRetriever

__all__ = [
    # Base
    "BaseRetriever",
    # Fusion
    "FusionLayer",
    # Reranking
    "cross_encoder_rerank",
    # Local retrieval
    "LocalRetriever",
    "BM25Index",
    "get_or_create_bm25_index",
    "get_or_create_vector_store",
    # Web retrieval
    "WebRetriever",
]
