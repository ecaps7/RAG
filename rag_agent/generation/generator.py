"""LLM-based answer generator."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Iterator

from ..config import get_config, init_model_with_config
from ..core.types import Answer, ContextChunk
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step
from .prompts import ANSWER_SYSTEM_PROMPT, ANSWER_STREAM_PROMPT


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parsing from model output."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            candidate = m.group(0)
            return json.loads(candidate)
    except Exception:
        pass
    try:
        fixed = re.sub(r"'(\s*[:\]},])", r'"\1', text)
        fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        return None


def _extract_text_content(content: Any) -> str:
    """Extract text from various LangChain output formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, dict):
                t = part.get("text") or part.get("content") or ""
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


class AnswerGenerator:
    """LLM-based answer generator with fallback support.
    
    Features:
    - Strict JSON output for structured answers
    - Streaming support for real-time output
    - Template fallback when LLM is unavailable
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        try:
            cfg = get_config()
            self._model = init_model_with_config(
                cfg.response_model_name,
                cfg.response_model_temperature
            )
        except Exception as e:
            self.logger.info("AnswerGenerator LLM init failed; will use fallback. (%s)", e)
            self._model = None

    def _format_contexts(self, chunks: List[ContextChunk]) -> List[Dict[str, Any]]:
        """Format chunks as context list for LLM."""
        ctxs: List[Dict[str, Any]] = []
        for ch in chunks:
            cite = ch.citation or ch.title or ch.source_id
            content = (ch.content or "").strip().replace("\n", " ")
            if len(content) > 1200:
                content = content[:1200] + "..."
            ctxs.append({
                "id": ch.id,
                "source_type": ch.source_type,
                "source_id": ch.source_id,
                "title": ch.title or "",
                "citation": cite or "",
                "score": ch.similarity,
                "similarity": ch.similarity,
                "reliability": ch.reliability,
                "recency": ch.recency,
                "content": content,
            })
        return ctxs

    def _fallback_answer(self, question: str, chunks: List[ContextChunk]) -> Answer:
        """Generate fallback answer when LLM is unavailable."""
        citations: List[str] = []
        parts: List[str] = []
        
        for i, ch in enumerate(chunks, start=1):
            cite = ch.citation or ch.title or ch.source_id
            if cite:
                citations.append(str(cite))
            snippet = (ch.content or "").strip().replace("\n", " ")
            parts.append(f"证据{i}（{ch.source_type}）: {snippet}")

        body = (
            f"问题：{question}\n"
            + ("\n".join(parts) if parts else "未检索到足够上下文，以下为概括性回答。")
            + "\n\n"
            + "综合以上证据，给出结论与解释（简要）："
        )

        conf = 0.0
        if chunks:
            conf = sum(ch.similarity for ch in chunks) / len(chunks)

        return Answer(text=body, citations=citations, confidence=conf, meta={"generator": "template"})

    @traceable_step("answer_generation", run_type="llm")
    def generate(self, question: str, chunks: List[ContextChunk]) -> Answer:
        """Generate an answer based on the question and retrieved chunks.
        
        Args:
            question: The user's question
            chunks: The retrieved context chunks
            
        Returns:
            Answer object with text, citations, and confidence
        """
        q = (question or "").strip()
        
        if not getattr(self, "_model", None):
            return self._fallback_answer(q, chunks)

        try:
            contexts = self._format_contexts(chunks)
            preview = {
                "question": q,
                "contexts": contexts,
                "output_format": {"answer": "...", "citations": ["..."], "confidence": 0.75},
            }
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(preview, ensure_ascii=False)},
            ]
            resp = self._model.invoke(messages)
            raw_text = _extract_text_content(getattr(resp, "content", resp)).strip()
            data = _safe_json_loads(raw_text)
        except Exception as e:
            self.logger.info("LLM generation failed; using fallback. (%s)", e)
            return self._fallback_answer(q, chunks)

        if not isinstance(data, dict):
            self.logger.info("LLM returned non-JSON; using fallback.")
            return self._fallback_answer(q, chunks)

        # Extract fields
        text = str(data.get("answer", "")).strip()
        cits_raw = data.get("citations") or []
        citations: List[str] = []
        if isinstance(cits_raw, list):
            citations = [str(c) for c in cits_raw if c]
        else:
            citations = [str(cits_raw)] if cits_raw else []
        
        if not citations:
            citations = [
                (ch.citation or ch.title or ch.source_id)
                for ch in chunks
                if (ch.citation or ch.title or ch.source_id)
            ]

        try:
            conf = float(data.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf <= 0.0:
            conf = sum(ch.similarity for ch in chunks) / (len(chunks) or 1)
        conf = max(0.0, min(1.0, conf))

        if not text:
            return self._fallback_answer(q, chunks)

        return Answer(text=text, citations=citations, confidence=conf, meta={"generator": "llm"})

    def stream_answer_text(self, question: str, chunks: List[ContextChunk]) -> Iterator[str]:
        """Stream the answer text token by token.
        
        Args:
            question: The user's question
            chunks: The retrieved context chunks
            
        Yields:
            Text tokens as they are generated
        """
        q = (question or "").strip()

        if not getattr(self, "_model", None):
            fallback = self._fallback_answer(q, chunks)
            parts = [p for p in re.split(r"(?<=[。！？\.!?])\s+|\n+", fallback.text) if p]
            for i, p in enumerate(parts):
                yield (p + ("\n" if i < len(parts) - 1 else ""))
            return

        try:
            contexts = self._format_contexts(chunks)
            preview = {"question": q, "contexts": contexts}
            messages = [
                {"role": "system", "content": ANSWER_STREAM_PROMPT},
                {"role": "user", "content": json.dumps(preview, ensure_ascii=False)},
            ]
            for chunk in self._model.stream(messages):
                delta = _extract_text_content(getattr(chunk, "content", chunk))
                if not delta:
                    continue
                yield delta
        except Exception as e:
            self.logger.info("LLM streaming failed; using fallback. (%s)", e)
            fallback = self._fallback_answer(q, chunks)
            parts = [p for p in re.split(r"(?<=[。！？\.!?])\s+|\n+", fallback.text) if p]
            for i, p in enumerate(parts):
                yield (p + ("\n" if i < len(parts) - 1 else ""))
