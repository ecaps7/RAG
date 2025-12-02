from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..common.logging import get_logger
from ..common.config import get_config, init_model_with_config


# 系统提示：严格的意图分类器（不含示例）
SYSTEM_PROMPT = (
    "你是一个严格的意图分类器。将用户问题归到以下之一：\n"
    "- data_lookup\n"
    "- definition_lookup\n"
    "- reasoning\n"
    "- external_context\n"
    "- forecast\n"
    "- meta_query\n\n"
    "规则：\n"
    "1) 只输出JSON，不要额外文字。\n"
    "2) 如果问题主要是问“是多少/同比/环比/披露/单位/口径/表格字段”，优先归为 data_lookup 或 definition_lookup。\n"
    "3) 如果涉及“原因/为什么/影响/对比解释”，优先 reasoning；若明显涉及宏观政策/行业，则 external_context。\n"
    "4) 询问未来或预计→ forecast。\n"
    "5) 关于文档本身（是否披露、在哪一章、页码等）→ meta_query。\n\n"
    "输出字段：intent, confidence(0-1), rationales(简短数组), need_web(bool), need_local(bool)。"
)

# 用户提示：包含示例的 few-shot 指引
USER_PROMPT = (
    "示例：\n"
    "Q: “2025年上半年净利息收入是多少？”\n"
    "A: {\"intent\":\"data_lookup\",\"confidence\":0.92,\"rationales\":[\"具体数值\"],\"need_web\":false,\"need_local\":true}\n\n"
    "Q: “净利差的计算口径是什么？”\n"
    "A: {\"intent\":\"definition_lookup\",\"confidence\":0.90,\"rationales\":[\"定义/口径\"],\"need_web\":false,\"need_local\":true}\n\n"
    "Q: “2025Q2 的净息差较 Q1 下降的原因是什么？”\n"
    "A: {\"intent\":\"reasoning\",\"confidence\":0.86,\"rationales\":[\"跨期对比+原因\"],\"need_web\":true,\"need_local\":true}\n\n"
    "Q: “LPR 调整对净息差有何影响？”\n"
    "A: {\"intent\":\"external_context\",\"confidence\":0.87,\"rationales\":[\"宏观政策\"],\"need_web\":true,\"need_local\":false}\n\n"
    "Q: “下半年净息差大概率如何变化？”\n"
    "A: {\"intent\":\"forecast\",\"confidence\":0.84,\"rationales\":[\"未来预期\"],\"need_web\":true,\"need_local\":false}\n\n"
    "Q: “报告是否披露资本充足率的高级法口径？”\n"
    "A: {\"intent\":\"meta_query\",\"confidence\":0.88,\"rationales\":[\"披露与否\"],\"need_web\":false,\"need_local\":true}\n"
)


ALLOWED_INTENTS = {
    "data_lookup",
    "definition_lookup",
    "reasoning",
    "external_context",
    "forecast",
    "meta_query",
}


def _infer_flags_from_intent(intent: str) -> Dict[str, bool]:
    intent = (intent or "").strip()
    if intent in {"data_lookup", "definition_lookup", "meta_query"}:
        return {"need_local": True, "need_web": False}
    if intent in {"external_context", "forecast"}:
        return {"need_local": False, "need_web": True}
    # reasoning 默认双路
    return {"need_local": True, "need_web": True}


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """尽力解析模型输出为 JSON 对象。若失败返回 None。"""
    if not text:
        return None
    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass
    # 提取第一个花括号包裹的对象
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            candidate = m.group(0)
            return json.loads(candidate)
    except Exception:
        pass
    # 尝试替换单引号为双引号
    try:
        fixed = re.sub(r"'(\s*[:\]},])", r'"\1', text)
        fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        return None


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


class LLMIntentClassifier:
    """基于 LLM 的意图分类器（只用 LLM，JSON-only 输出）。

    使用 rag_agent.common.config.init_model_with_config 初始化模型，并按系统提示生成严格 JSON。
    - 仅 LLM 路径：初始化失败或解析失败将抛出异常；
    - 始终校正 need_web/need_local 与 intent 一致；
    - 置信度在 [0,1] 范围内并保留两位精度。"""

    def __init__(self, trace_id: Optional[str] = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        try:
            cfg = get_config()
            self._model = init_model_with_config(cfg.response_model_name, cfg.response_model_temperature)
        except Exception as e:
            raise RuntimeError(f"LLMIntentClassifier model init failed: {e}")

    def classify(self, question: str) -> Dict[str, Any]:
        """返回包含 intent, confidence, rationales, need_web, need_local 的字典。"""
        q = (question or "").strip()

        # 仅 LLM：生成严格 JSON
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
            raise RuntimeError(f"LLM classify failed: {e}")

        if not isinstance(result, dict):
            raise ValueError(f"LLM output is not valid JSON: {raw_text}")

        # 3) 校验与修正字段
        intent = str(result.get("intent", "reasoning")).strip()
        if intent not in ALLOWED_INTENTS:
            intent = "reasoning"
        try:
            conf = float(result.get("confidence", 0.6))
        except Exception:
            conf = 0.6
        conf = max(0.0, min(1.0, conf))

        rationals: List[str] = result.get("rationales") or []
        if not isinstance(rationals, list):
            rationals = [str(rationals)]
        rationals = [str(r)[:20] for r in rationals if r is not None]
        if not rationals:
            # 简短自动 rationale
            auto_r = {
                "data_lookup": "具体数值/字段",
                "definition_lookup": "定义/口径",
                "reasoning": "原因/解释",
                "external_context": "宏观/行业",
                "forecast": "未来预期",
                "meta_query": "文档披露/页码",
            }[intent]
            rationals = [auto_r]

        # 依据 intent 统一 need_web/need_local
        flags = _infer_flags_from_intent(intent)
        need_web = bool(result.get("need_web", flags["need_web"]))
        need_local = bool(result.get("need_local", flags["need_local"]))

        return {
            "intent": intent,
            "confidence": round(conf, 2),
            "rationales": rationals,
            "need_web": need_web,
            "need_local": need_local,
        }

    def classify_json(self, question: str) -> str:
        """返回严格 JSON 字符串（不含额外文字）。"""
        result = self.classify(question)
        return json.dumps(result, ensure_ascii=False)