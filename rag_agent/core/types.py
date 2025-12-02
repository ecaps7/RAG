"""Core data types for the RAG agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional


class Intent(str, Enum):
    """User intent classification categories."""
    data_lookup = "data_lookup"
    definition_lookup = "definition_lookup"
    reasoning = "reasoning"
    external_context = "external_context"
    forecast = "forecast"
    meta_query = "meta_query"


SourceType = Literal["local", "web"]


@dataclass
class ContextChunk:
    """A chunk of context retrieved from local or web sources."""
    id: str
    content: str
    source_type: SourceType
    source_id: str
    title: Optional[str] = None
    similarity: float = 0.0
    recency: float = 0.5
    reliability: float = 0.5
    metadata: Dict[str, str] = field(default_factory=dict)
    citation: Optional[str] = None


@dataclass
class RetrievalPlan:
    """Plan for retrieval based on intent classification."""
    use_local: bool
    use_web: bool
    local_top_k: int = 5
    web_top_k: int = 5
    hybrid_strategy: Literal["merge", "interleave", "balance"] = "balance"


@dataclass
class FusionResult:
    """Result of fusing chunks from multiple sources."""
    selected_chunks: List[ContextChunk]
    scores: Dict[str, float]


@dataclass
class Answer:
    """Final answer with citations and confidence."""
    text: str
    citations: List[str]
    confidence: float
    meta: Dict[str, str] = field(default_factory=dict)
