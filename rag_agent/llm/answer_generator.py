from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Iterator

from ..common.logging import get_logger
from ..common.config import get_config, init_model_with_config
from ..common.types import Answer, FusionResult


# 系统提示：严格 JSON 输出的回答生成器
SYSTEM_PROMPT = (
    "你是一个严谨的报告问答助手。只根据提供的 contexts 回答问题。\n"
    "规则：\n"
    "1) 只输出JSON，不要额外文字。\n"
    "2) 不要臆测，结论应源于 contexts；若证据不足请明确说明。\n"
    "3) 用中文作答，结构清晰（先给结论，再简要解释）。\n"
    "4) citations 仅取自每个 context 的 citation/title/source_id，避免重复。\n\n"
    "输出字段：answer(字符串)、citations(数组字符串)、confidence(0-1)。"
)

# 流式输出提示：仅输出中文答案文本，不含任何JSON或标签
SYSTEM_PROMPT_STREAM = (
    "你是一个严谨的报告问答助手。只根据提供的 contexts 回答问题。\n"
    "规则：\n"
    "1) 只输出中文答案文本，不要JSON，也不要附加标签。\n"
    "2) 不要臆测，结论应源于 contexts；若证据不足请明确说明。\n"
    "3) 结构清晰：先给结论，再简要解释；避免逐字复制原文，做归纳表达。\n"
)


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """尽力解析模型输出为 JSON 对象。若失败返回 None。"""
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
    """兼容 LangChain chat model 的内容抽取（list/dict/str）。"""
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
    """LLM-based answer generator with robust fallback.

    - 使用 rag_agent.common.config.init_model_with_config 初始化聊天模型；
    - 以严格 JSON-only 的协议让模型输出 answer/citations/confidence；
    - 解析失败或模型不可用时，回退到模板化组合答案。
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        try:
            cfg = get_config()
            self._model = init_model_with_config(cfg.response_model_name, cfg.response_model_temperature)
        except Exception as e:
            # 与 web/local 检索器一致：初始化失败则记录并允许回退
            self.logger.info("AnswerGenerator LLM init failed; will use template fallback. (%s)", e)
            self._model = None

    def _format_contexts(self, fusion: FusionResult) -> List[Dict[str, Any]]:
        ctxs: List[Dict[str, Any]] = []
        for ch in fusion.selected_chunks:
            cite = ch.citation or ch.title or ch.source_id
            content = (ch.content or "").strip().replace("\n", " ")
            # 控制长度，避免提示过长
            if len(content) > 1200:
                content = content[:1200] + "..."
            ctxs.append({
                "id": ch.id,
                "source_type": ch.source_type,
                "source_id": ch.source_id,
                "title": ch.title or "",
                "citation": cite or "",
                "score": float(fusion.scores.get(ch.id, 0.0)),
                "similarity": float(getattr(ch, "similarity", 0.0) or 0.0),
                "reliability": float(getattr(ch, "reliability", 0.5) or 0.5),
                "recency": float(getattr(ch, "recency", 0.5) or 0.5),
                "content": content,
            })
        return ctxs

    def _fallback_answer(self, question: str, fusion: FusionResult) -> Answer:
        chunks = fusion.selected_chunks
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
            conf = sum(fusion.scores.get(ch.id, 0.0) for ch in chunks) / len(chunks)

        return Answer(text=body, citations=citations, confidence=conf, meta={"generator": "template"})

    def generate(self, question: str, fusion: FusionResult) -> Answer:
        q = (question or "").strip()
        # 若模型不可用，直接回退
        if not getattr(self, "_model", None):
            return self._fallback_answer(q, fusion)

        # 组装上下文并调用 LLM（严格 JSON-only）
        try:
            contexts = self._format_contexts(fusion)
            preview = {
                "question": q,
                "contexts": contexts,
                "output_format": {"answer": "...", "citations": ["..."], "confidence": 0.75},
            }
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(preview, ensure_ascii=False)},
            ]
            resp = self._model.invoke(messages)
            raw_text = _extract_text_content(getattr(resp, "content", resp)).strip()
            data = _safe_json_loads(raw_text)
        except Exception as e:
            self.logger.info("LLM answer generation failed; using fallback. (%s)", e)
            return self._fallback_answer(q, fusion)

        if not isinstance(data, dict):
            self.logger.info("LLM returned non-JSON content; using fallback. Raw=%.120s", raw_text)
            return self._fallback_answer(q, fusion)

        # 提取字段并校验
        text = str(data.get("answer", "")).strip()
        cits_raw = data.get("citations") or []
        citations: List[str] = []
        if isinstance(cits_raw, list):
            citations = [str(c) for c in cits_raw if c]
        else:
            citations = [str(cits_raw)] if cits_raw else []
        # 若模型未给 citations，则从上下文补全
        if not citations:
            citations = [
                (ch.citation or ch.title or ch.source_id)
                for ch in fusion.selected_chunks
                if (ch.citation or ch.title or ch.source_id)
            ]

        # 置信度处理：优先 LLM，回退为融合分数均值
        try:
            conf = float(data.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf <= 0.0:
            chunks = fusion.selected_chunks
            conf = sum(fusion.scores.get(ch.id, 0.0) for ch in chunks) / (len(chunks) or 1)
        conf = max(0.0, min(1.0, conf))

        # 若模型文本为空，则回退模板；否则返回 LLM 结果
        if not text:
            return self._fallback_answer(q, fusion)

        return Answer(text=text, citations=citations, confidence=conf, meta={"generator": "llm"})

    def stream_answer_text(self, question: str, fusion: FusionResult) -> Iterator[str]:
        """以流式方式生成最终答案文本。

        - 当模型可用时，使用聊天模型的 .stream 接口基于 contexts 生成纯文本答案；
        - 当模型不可用时，回退到模板化答案并按句子切分流式输出；
        """
        q = (question or "").strip()

        # 回退：模型不可用
        if not getattr(self, "_model", None):
            fallback = self._fallback_answer(q, fusion)
            parts = [p for p in re.split(r"(?<=[。！？\.!?])\s+|\n+", fallback.text) if p]
            for i, p in enumerate(parts):
                yield (p + ("\n" if i < len(parts) - 1 else ""))
            return

        # 正常：使用模型流式生成纯文本答案
        try:
            contexts = self._format_contexts(fusion)
            preview = {"question": q, "contexts": contexts}
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_STREAM},
                {"role": "user", "content": json.dumps(preview, ensure_ascii=False)},
            ]
            for chunk in self._model.stream(messages):
                delta = _extract_text_content(getattr(chunk, "content", chunk))
                if not delta:
                    continue
                yield delta
        except Exception as e:
            self.logger.info("LLM streaming failed; using fallback. (%s)", e)
            fallback = self._fallback_answer(q, fusion)
            parts = [p for p in re.split(r"(?<=[。！？\.!?])\s+|\n+", fallback.text) if p]
            for i, p in enumerate(parts):
                yield (p + ("\n" if i < len(parts) - 1 else ""))