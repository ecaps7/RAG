"""Custom exceptions for the RAG agent."""

from __future__ import annotations


class RagAgentError(Exception):
    """Base exception for RAG agent errors."""
    pass


class ConfigurationError(RagAgentError):
    """Raised when there's a configuration error."""
    pass


class RetrievalError(RagAgentError):
    """Raised when retrieval fails."""
    pass


class GenerationError(RagAgentError):
    """Raised when answer generation fails."""
    pass


class ClassificationError(RagAgentError):
    """Raised when intent classification fails."""
    pass
