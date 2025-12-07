"""Core types, interfaces, and exceptions for rag_agent."""

from .types import (
    SourceType,
    ContextChunk,
    Answer,
    CitationInfo,
)
from .interfaces import Retriever, Generator
from .exceptions import (
    RagAgentError,
    ConfigurationError,
    RetrievalError,
    GenerationError,
    ClassificationError,
)

__all__ = [
    # Types
    "SourceType",
    "ContextChunk",
    "Answer",
    "CitationInfo",
    # Interfaces
    "Retriever",
    "Generator",
    # Exceptions
    "RagAgentError",
    "ConfigurationError",
    "RetrievalError",
    "GenerationError",
    "ClassificationError",
]
