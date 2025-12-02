"""Abstract interfaces (Protocols) for RAG agent components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Protocol, Tuple, Any

from .types import Answer, ContextChunk, FusionResult, Intent


class Retriever(Protocol):
    """Protocol for retriever implementations."""
    
    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        """Retrieve relevant chunks for the given query."""
        ...


class IntentClassifierProtocol(Protocol):
    """Protocol for intent classifier implementations."""
    
    def classify(self, question: str) -> Tuple[Intent, float]:
        """Classify the intent of a question.
        
        Returns:
            Tuple of (Intent, confidence_score)
        """
        ...


class Generator(Protocol):
    """Protocol for answer generator implementations."""
    
    def generate(self, question: str, fusion: FusionResult) -> Answer:
        """Generate an answer based on the question and fused context."""
        ...
    
    def stream_answer_text(self, question: str, fusion: FusionResult) -> Iterator[str]:
        """Stream the answer text token by token."""
        ...


class FusionLayerProtocol(Protocol):
    """Protocol for fusion layer implementations."""
    
    def aggregate(
        self,
        local: List[ContextChunk],
        web: List[ContextChunk],
        intent: Intent,
        k: int | None = None,
    ) -> FusionResult:
        """Aggregate and rank chunks from multiple sources."""
        ...


class RetrievalRouterProtocol(Protocol):
    """Protocol for retrieval router implementations."""
    
    def plan(self, intent: Intent) -> "RetrievalPlan":
        """Create a retrieval plan based on intent."""
        ...


# Import here to avoid circular imports
from .types import RetrievalPlan
