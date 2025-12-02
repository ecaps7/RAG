"""Constant values for the RAG agent."""

from __future__ import annotations

from typing import Dict


# Source reliability scores
SOURCE_RELIABILITY: Dict[str, float] = {
    "local": 0.9,
    "web": 0.6,
}


# Top-K retrieval settings
TOP_K: Dict[str, int] = {
    "local": 5,
    "web": 5,
    "fusion": 6,
}


# Recency decay configuration (used by fusion layer utilities)
RECENCY_HALFLIFE_DAYS = 365  # one year
