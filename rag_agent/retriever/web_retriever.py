from __future__ import annotations

from typing import List
import urllib.parse as up

from ..common.logging import get_logger
from ..common.types import ContextChunk


class WebRetriever:
    """Web retriever implemented fully within rag_agent.

    Uses rag_agent.utils.websearch.make_query + search (Tavily) to fetch results,
    computes similarity via rag_agent.common.utils.compute_overlap_ratio,
    and reliability via cfg.source_reliability['web'] domain mapping.

    When Tavily is unavailable or invocation fails, the error is propagated
    (fail fast) to allow upstream handling.
    """

    def __init__(self, trace_id: str | None = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        try:
            from ..utils.websearch import make_query, search  # type: ignore
            from ..common.utils import compute_overlap_ratio  # type: ignore
            from ..common.config import get_config  # type: ignore
            self._make_query = make_query
            self._search = search
            self._compute_overlap = compute_overlap_ratio
            self._get_cfg = get_config
            self._available = True
        except Exception as e:
            self.logger.info("Web retriever init failed; will use empty results. (%s)", e)
            self._available = False

    def _domain_from_url(self, url: str) -> str:
        try:
            return (up.urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    def _reliability_for_domain(self, domain: str) -> float:
        cfg = self._get_cfg()
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
        if not getattr(self, "_available", False):
            self.logger.info("Web search not available; returning empty list.")
            return []

        # Generate search query and perform Tavily search. Exceptions from
        # the underlying websearch.search are intentionally propagated.
        q = self._make_query(query)
        results = self._search(q)
        cfg = self._get_cfg()
        chunks: List[ContextChunk] = []
        for i, r in enumerate(results[:top_k]):
            domain = self._domain_from_url(r.url)
            rel = self._reliability_for_domain(domain)
            sim = float(self._compute_overlap(query, r.content, cfg.zh_stopwords) or 0.0)
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
                    recency=0.7,  # basic default; can be improved with date extraction
                    metadata={"domain": domain, "score": str(r.score)},
                    citation=citation,
                )
            )
        self.logger.info("Web retrieved %d chunks via Tavily", len(chunks))
        return chunks