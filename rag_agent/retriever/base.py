from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..common.types import ContextChunk


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        raise NotImplementedError