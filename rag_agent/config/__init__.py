"""Configuration module for rag_agent."""

from .settings import AppConfig, get_config, init_model_with_config
from .constants import SOURCE_RELIABILITY, TOP_K, RECENCY_HALFLIFE_DAYS
from .weights import DEFAULT_WEIGHTS, INTENT_WEIGHTS, get_weights

__all__ = [
    # Settings
    "AppConfig",
    "get_config",
    "init_model_with_config",
    # Constants
    "SOURCE_RELIABILITY",
    "TOP_K",
    "RECENCY_HALFLIFE_DAYS",
    # Weights
    "DEFAULT_WEIGHTS",
    "INTENT_WEIGHTS",
    "get_weights",
]
