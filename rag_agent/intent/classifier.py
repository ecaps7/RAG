"""LLM-based intent classifier."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_config, init_model_with_config
from ..core.types import Intent
from ..core.exceptions import ClassificationError
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step


# System prompt for strict JSON output
SYSTEM_PROMPT = """你是一个严格的意图分类器。将用户问题归到以下之一：
- data_lookup
- definition_lookup
- reasoning
- external_context
- forecast
- meta_query

规则：
1) 只输出JSON，不要额外文字。
2) 如果问题主要是问是多少/同比/环比/披露/单位/口径/表格字段，优先归为 data_lookup 或 definition_lookup。
3) 如果涉及原因/为什么/影响/对比解释，优先 reasoning；若明显涉及宏观政策/行业，则 external_context。
4) 询问未来或预计 -> forecast。
5) 关于文档本身（是否披露、在哪一章、页码等）-> meta_query。

输出字段：intent, confidence(0-1), rationales(简短数组), need_web(bool), need_local(bool)。"""

# Few-shot examples
USER_PROMPT = """示例：
Q: "2025年上半年净利息收入是多少?"
A: {"intent":"data_lookup","confidence":0.92,"rationales":["具体数值"],"need_web":false,"need_local":true}

Q: "净利差的计算口径是什么?"
A: {"intent":"definition_lookup","confidence":0.90,"rationales":["定义/口径"],"need_web":false,"need_local":true}

Q: "2025Q2 的净息差较 Q1 下降的原因是什么?"
A: {"intent":"reasoning","confidence":0.86,"rationales":["跨期对比+原因"],"need_web":true,"need_local":true}

Q: "LPR 调整对净息差有何影响?"
A: {"intent":"external_context","confidence":0.87,"rationales":["宏观政策"],"need_web":true,"need_local":false}

Q: "下半年净息差大概率如何变化?"
A: {"intent":"forecast","confidence":0.84,"rationales":["未来预期"],"need_web":true,"need_local":false}

Q: "报告是否披露资本充足率的高级法口径?"
A: {"intent":"meta_query","confidence":0.88,"rationales":["披露与否"],"need_web":false,"need_local":true}
"""


ALLOWED_INTENTS = {
    "data_lookup",
    "definition_lookup",
    "reasoning",
    "external_context",
    "forecast",
    "meta_query",
}


def _infer_flags_from_intent(intent: str) -> Dict[str, bool]:
    """Infer need_web and need_local flags from intent."""
    intent = (intent or "").strip()
    if intent in {"data_lookup", "definition_lookup", "meta_query"}:
        return {"need_local": True, "need_web": False}
    if intent in {"external_context", "forecast"}:
        return {"need_local": False, "need_web": True}
    # reasoning defaults to both
    return {"need_local": True, "need_web": True}


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parsing from model output."""
    if not text:
        return None
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first JSON object
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            candidate = m.group(0)
            return json.loads(candidate)
    except Exception:
        pass
    # Try fixing quotes
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
        return "".join(parts)
    return str(content)


class IntentClassifier:
    """LLM-based intent classifier.
    
    Classifies user questions into predefined intents and returns
    a tuple of (Intent, confidence_score).
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self._trace_id = trace_id or ""
        try:
            cfg = get_config()
            self._model = init_model_with_config(
                cfg.response_model_name,
                cfg.response_model_temperature
            )
        except Exception as e:
            raise ClassificationError(f"Failed to initialize intent classifier model: {e}")

    def _classify_raw(self, question: str) -> Dict[str, Any]:
        """Get raw classification result from LLM."""
        q = (question or "").strip()

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
                {"role": "user", "content": f"现在对这个问题分类：\nQ: \"{q}\""},
            ]
            response = self._model.invoke(messages)
            raw_text = _extract_text_content(response.content).strip()
            result = _safe_json_loads(raw_text)
        except Exception as e:
            raise ClassificationError(f"LLM classification failed: {e}")

        if not isinstance(result, dict):
            raise ClassificationError(f"LLM output is not valid JSON: {raw_text}")

        # Validate and correct fields
        intent = str(result.get("intent", "reasoning")).strip()
        if intent not in ALLOWED_INTENTS:
            intent = "reasoning"

        try:
            conf = float(result.get("confidence", 0.6))
        except Exception:
            conf = 0.6
        conf = max(0.0, min(1.0, conf))

        rationales: List[str] = result.get("rationales") or []
        if not isinstance(rationales, list):
            rationales = [str(rationales)]
        rationales = [str(r)[:20] for r in rationales if r is not None]
        if not rationales:
            auto_rationale = {
                "data_lookup": "具体数值/字段",
                "definition_lookup": "定义/口径",
                "reasoning": "原因/解释",
                "external_context": "宏观/行业",
                "forecast": "未来预期",
                "meta_query": "文档披露/页码",
            }[intent]
            rationales = [auto_rationale]

        # Infer flags from intent
        flags = _infer_flags_from_intent(intent)
        need_web = bool(result.get("need_web", flags["need_web"]))
        need_local = bool(result.get("need_local", flags["need_local"]))

        return {
            "intent": intent,
            "confidence": round(conf, 2),
            "rationales": rationales,
            "need_web": need_web,
            "need_local": need_local,
        }

    @traceable_step("intent_classification", run_type="chain")
    def classify(self, question: str) -> Tuple[Intent, float]:
        """Classify the intent of a question.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (Intent enum, confidence score)
        """
        q = (question or "").strip()
        if not q:
            return (Intent.reasoning, 0.0)

        data = self._classify_raw(q)

        intent_str = str(data.get("intent", "reasoning")).strip().lower()
        try:
            intent = Intent(intent_str)
        except Exception:
            raise ClassificationError(f"Invalid intent from LLM: {intent_str}")

        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        self.logger.info("Intent=%s conf=%.2f", intent.value, conf)
        return intent, conf

    def classify_json(self, question: str) -> str:
        """Return classification result as JSON string."""
        result = self._classify_raw(question)
        return json.dumps(result, ensure_ascii=False)
