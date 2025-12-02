from __future__ import annotations

from typing import Tuple

from ..common.logging import get_logger
from ..common.types import Intent
from ..llm.classifier_llm import LLMIntentClassifier


class IntentClassifier:
    """适配层：直接调用 rag_agent.llm.classifier_llm.LLMIntentClassifier。

    保持现有接口：classify(question) -> (Intent, confidence)
    """

    def __init__(self, trace_id: str | None = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self._trace_id = trace_id or ""
        self._llm = LLMIntentClassifier(trace_id=self._trace_id)

    def classify(self, question: str) -> Tuple[Intent, float]:
        q = (question or "").strip()
        if not q:
            return (Intent.reasoning, 0.0)

        data = self._llm.classify(q)

        intent_str = str(data.get("intent", "reasoning")).strip().lower()
        try:
            intent = Intent(intent_str)
        except Exception:
            raise ValueError(f"Invalid intent from LLM: {intent_str}")

        conf_val = data.get("confidence", 0.0)
        try:
            conf = float(conf_val)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        self.logger.info("Intent(LLM)=%s conf=%.2f", intent.value, conf)
        return intent, conf