"""Base retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core.types import ContextChunk


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        """Retrieve relevant chunks for the given query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            List of ContextChunk objects
        """
        raise NotImplementedError
