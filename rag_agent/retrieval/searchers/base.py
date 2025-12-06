from abc import ABC, abstractmethod
from typing import List
from ..types import SearchResult

class BaseSearcher(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        pass