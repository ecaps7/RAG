"""Web retriever using Tavily search."""

from __future__ import annotations

from typing import List
import urllib.parse as up

from ...config import get_config
from ...core.types import ContextChunk
from ...utils.logging import get_logger
from ...utils.text import compute_overlap_ratio


class WebRetriever:
    """Web retriever using Tavily search API.
    
    Computes similarity via word overlap and reliability via domain mapping.
    """

    def __init__(self, trace_id: str | None = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        
        try:
            from ...utils.websearch import make_query, search
            self._make_query = make_query
            self._search = search
            self._available = True
        except Exception as e:
            self.logger.info("Web retriever init failed: %s", e)
            self._available = False

    def _domain_from_url(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            return (up.urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    def _reliability_for_domain(self, domain: str) -> float:
        """Get reliability score for a domain."""
        cfg = get_config()
        web_map = (cfg.source_reliability or {}).get("web", {})
        
        for key, val in web_map.items():
            if key == "default":
                continue
            if domain.endswith(key):
                try:
                    return float(val)
                except Exception:
                    continue
        
        try:
            return float(web_map.get("default", 0.60))
        except Exception:
            return 0.60

    def retrieve(self, query: str, top_k: int) -> List[ContextChunk]:
        """Retrieve relevant chunks from web search.
        
        Args:
            query: The search query
            top_k: Maximum number of results
            
        Returns:
            List of ContextChunk objects
        """
        if not getattr(self, "_available", False):
            self.logger.info("Web search not available; returning empty list.")
            return []

        q = self._make_query(query)
        results = self._search(q)
        cfg = get_config()
        
        chunks: List[ContextChunk] = []
        for i, r in enumerate(results[:top_k]):
            domain = self._domain_from_url(r.url)
            rel = self._reliability_for_domain(domain)
            sim = float(compute_overlap_ratio(query, r.content, cfg.zh_stopwords) or 0.0)
            citation = f"{r.title} - {r.url}" if r.url else r.title
            
            chunks.append(
                ContextChunk(
                    id=f"web-{i}",
                    content=r.content,
                    source_type="web",
                    source_id=r.url or (r.title or f"web#{i}"),
                    title=r.title or None,
                    similarity=sim,
                    reliability=rel,
                    recency=0.7,  # Default; can be improved with date extraction
                    metadata={"domain": domain, "score": str(r.score)},
                    citation=citation,
                )
            )
        
        self.logger.info("Web retrieved %d chunks", len(chunks))
        return chunks
