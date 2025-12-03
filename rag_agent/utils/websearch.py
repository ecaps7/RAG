from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any

from ..config import get_config, init_model_with_config

# Tavily tool integration
try:
    from langchain_tavily import TavilySearch  # type: ignore
except Exception:
    TavilySearch = None  # type: ignore


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


_SEARCH_QUERY_PROMPT = """你是搜索查询优化专家。将问题转换为搜索关键词。

原始问题: {question}

要求:
- 保留实体和时间词
- 移除疑问词
- 5-8个关键词
- 用空格分隔

搜索查询:"""


def _get_response_model():
    cfg = get_config()
    try:
        return init_model_with_config(cfg.response_model_name, cfg.response_model_temperature)
    except Exception:
        return None


def _extract_text_content(content: Any) -> str:
    """Best-effort extraction of textual content from model outputs.

    Handles LangChain chat model outputs that may return a list of dicts
    or a plain string. Returns a stripped string.
    """
    if content is None:
        return ""
    try:
        if isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("text") or c.get("content") or ""
                    if isinstance(t, str):
                        parts.append(t)
                elif isinstance(c, str):
                    parts.append(c)
                else:
                    parts.append(str(c))
            return "".join(parts).strip()
        return str(content).strip()
    except Exception:
        return str(content) if content is not None else ""


def make_query(question: str) -> str:
    """Generate a search query from the question.

    Uses configured chat model when available; otherwise returns the original question.
    """
    q = (question or "").strip()
    model = _get_response_model()
    if model is None:
        return q
    try:
        prompt = _SEARCH_QUERY_PROMPT.format(question=q)
        resp = model.invoke([{"role": "user", "content": prompt}])
        text = _extract_text_content(getattr(resp, "content", resp))
        # If the model echoes a label, strip it
        if "搜索查询:" in text:
            text = text.split("搜索查询:", 1)[1].strip()
        return text or q
    except Exception:
        return q


def _get_tavily_tool():
    if TavilySearch is None:
        return None
    try:
        return TavilySearch(max_results=5, topic="general")
    except Exception:
        return None


def search(query: str) -> List[SearchResult]:
    """Run Tavily search and map results into SearchResult list.

    When Tavily is unavailable or invocation fails, raise RuntimeError to surface
    the error to callers (fail fast as requested).
    """
    tool = _get_tavily_tool()
    if tool is None:
        raise RuntimeError(
            "TavilySearch unavailable: ensure langchain-tavily is installed and TAVILY_API_KEY is set."
        )
    try:
        raw = tool.invoke({"query": query})
    except Exception as e:
        raise RuntimeError(f"TavilySearch invocation failed for query {query!r}: {e}")

    items: List[dict] = []
    if isinstance(raw, dict) and "results" in raw:
        items = list(raw.get("results", []))
    elif isinstance(raw, list):
        # some versions may return a list directly
        items = raw  # type: ignore

    results: List[SearchResult] = []
    for item in items[:5]:
        if isinstance(item, dict):
            title = item.get("title", "无标题")
            url = item.get("url", "")
            content = item.get("content", "")
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if content and len(str(content).strip()) > 10:
                results.append(SearchResult(title=title, url=url, content=content, score=score))
    return results