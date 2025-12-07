"""Abstract interfaces (Protocols) for RAG agent components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Protocol, Tuple, Any

from .types import Answer, ContextChunk


class Retriever(Protocol):
    """Protocol for retriever implementations."""
    
    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        """Retrieve relevant chunks for the given query."""
        ...


class Generator(Protocol):
    """Protocol for answer generator implementations."""
    
    def generate(self, question: str, chunks: List[ContextChunk]) -> Answer:
        """Generate an answer based on the question and retrieved chunks."""
        ...
    
    def stream_answer_text(self, question: str, chunks: List[ContextChunk]) -> Iterator[str]:
        """Stream the answer text token by token."""
        ...
