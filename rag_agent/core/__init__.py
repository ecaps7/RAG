"""Core types, interfaces, and exceptions for rag_agent."""

from .types import (
    Intent,
    SourceType,
    ContextChunk,
    RetrievalPlan,
    FusionResult,
    Answer,
)
from .interfaces import Retriever, IntentClassifierProtocol, Generator
from .exceptions import (
    RagAgentError,
    ConfigurationError,
    RetrievalError,
    GenerationError,
    ClassificationError,
)

__all__ = [
    # Types
    "Intent",
    "SourceType",
    "ContextChunk",
    "RetrievalPlan",
    "FusionResult",
    "Answer",
    # Interfaces
    "Retriever",
    "IntentClassifierProtocol",
    "Generator",
    # Exceptions
    "RagAgentError",
    "ConfigurationError",
    "RetrievalError",
    "GenerationError",
    "ClassificationError",
]
