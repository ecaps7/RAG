"""LLM-based answer generator."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Iterator

from ..config import get_config
from ..core.types import Answer, ContextChunk
from ..llm import llm_services, AnswerOutput
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step
from .prompts import ANSWER_SYSTEM_PROMPT



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
    _instance: Optional["AnswerGenerator"] = None
    _initialized: bool = False

    def __new__(cls, trace_id: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, trace_id: Optional[str] = None):
        if not AnswerGenerator._initialized:
            self.logger = get_logger(self.__class__.__name__, trace_id)
            try:
                # Use unified LLM service
                self._base_model = llm_services.get_model()
                # Use structured model for non-streaming calls
                self._structured_model = llm_services.get_structured_model(AnswerOutput)
                self.logger.info("AnswerGenerator LLM initialized successfully.")
                AnswerGenerator._initialized = True
            except Exception as e:
                self.logger.info("AnswerGenerator LLM init failed; will use fallback. (%s)", e)
                self._base_model = None
                self._structured_model = None
                AnswerGenerator._initialized = True

    def _format_contexts(self, chunks: List[ContextChunk]) -> List[Dict[str, Any]]:
        """Format chunks as context list for LLM with numbered references."""
        ctxs: List[Dict[str, Any]] = []
        for idx, ch in enumerate(chunks, start=1):
            cite = ch.citation or ch.title or ch.source_id
            content = (ch.content or "").strip().replace("\n", " ")
            if len(content) > 1200:
                content = content[:1200] + "..."
            
            # 标注高可信来源
            is_high_confidence = ch.similarity >= 0.9 or ch.source_type == "sql_database"
            
            ctxs.append({
                "ref": idx,  # 引用编号，供 LLM 使用 [n] 标记
                "source_type": ch.source_type,
                "title": ch.title or cite or "",
                "score": round(ch.similarity, 3),
                "high_confidence": is_high_confidence,  # 明确标注高可信来源
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
        
        if not self._structured_model:
            return self._fallback_answer(q, chunks)

        try:
            contexts = self._format_contexts(chunks)
            preview = {
                "question": q,
                "contexts": contexts,
            }
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": str(preview)},
            ]
            
            # Use structured model for reliable output
            resp = self._structured_model.invoke(messages)
            
            # Extract fields directly from structured output
            text = str(resp.answer).strip()
            citations = [str(c) for c in resp.citations if c]
            
            if not citations:
                citations = [
                    (ch.citation or ch.title or ch.source_id)
                    for ch in chunks
                    if (ch.citation or ch.title or ch.source_id)
                ]

            conf = resp.confidence
            if conf <= 0.0:
                conf = sum(ch.similarity for ch in chunks) / (len(chunks) or 1)
            conf = max(0.0, min(1.0, conf))

            if not text:
                return self._fallback_answer(q, chunks)

            return Answer(text=text, citations=citations, confidence=conf, meta={"generator": "llm"})
        except Exception as e:
            self.logger.info("LLM generation failed; using fallback. (%s)", e)
            return self._fallback_answer(q, chunks)

    def stream_answer_text(self, question: str, chunks: List[ContextChunk]) -> Iterator[str]:
        """Stream the answer text token by token.
        
        Args:
            question: The user's question
            chunks: The retrieved context chunks
            
        Yields:
            Text tokens as they are generated
        """
        q = (question or "").strip()

        if not self._base_model:
            fallback = self._fallback_answer(q, chunks)
            parts = [p for p in re.split(r"(?<=[。！？\.!?])\s+|\n+", fallback.text) if p]
            for i, p in enumerate(parts):
                yield (p + ("\n" if i < len(parts) - 1 else ""))
            return

        try:
            contexts = self._format_contexts(chunks)
            preview = {"question": q, "contexts": contexts}
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": str(preview)},
            ]
            for chunk in self._base_model.stream(messages):
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
