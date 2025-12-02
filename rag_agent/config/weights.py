"""Intent-based weight configurations for the fusion layer."""

from __future__ import annotations

from typing import Dict

from ..core.types import Intent


# Base weights for fusion layer
DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_sim": 0.6,
    "w_rel": 0.3,
    "w_rec": 0.1,
}


# Intent-specific weights
INTENT_WEIGHTS: Dict[Intent, Dict[str, float]] = {
    Intent.external_context: {"w_sim": 0.55, "w_rel": 0.2, "w_rec": 0.25},
    Intent.forecast: {"w_sim": 0.55, "w_rel": 0.2, "w_rec": 0.25},
    Intent.definition_lookup: {"w_sim": 0.4, "w_rel": 0.5, "w_rec": 0.1},
    Intent.meta_query: {"w_sim": 0.35, "w_rel": 0.55, "w_rec": 0.1},
    Intent.reasoning: {"w_sim": 0.55, "w_rel": 0.25, "w_rec": 0.20},
    Intent.data_lookup: DEFAULT_WEIGHTS,
}


def get_weights(intent: Intent) -> Dict[str, float]:
    """Get fusion weights for a given intent."""
    return INTENT_WEIGHTS.get(intent, DEFAULT_WEIGHTS)
