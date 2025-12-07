"""RAG Agent - A modular Retrieval-Augmented Generation system.

This package provides:
- Hybrid retrieval (BM25 + Vector search)
- Multi-source fusion with MMR diversity
- LLM-powered answer generation with citations
- Short-term memory for multi-turn conversations

Main entry point:
    from rag_agent import RagAgent
    
    agent = RagAgent()
    answer = agent.run("你的问题")
"""

# Suppress jieba's pkg_resources deprecation warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

from .agent import RagAgent
from .core.types import (
    SourceType,
    ContextChunk,
    Answer,
)
from .core.exceptions import (
    RagAgentError,
    ConfigurationError,
    RetrievalError,
    GenerationError,
    ClassificationError,
)

__version__ = "0.2.0"

__all__ = [
    # Main agent
    "RagAgent",
    # Types
    "SourceType",
    "ContextChunk",
    "Answer",
    # Exceptions
    "RagAgentError",
    "ConfigurationError",
    "RetrievalError",
    "GenerationError",
    "ClassificationError",
]