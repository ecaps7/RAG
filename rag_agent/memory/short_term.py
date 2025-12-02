from __future__ import annotations

from typing import Any, Dict, List

from ..common.config import get_config, init_model_with_config


_REWRITE_PROMPT = (
    "你是查询改写助手。根据以下多轮对话，重写用户最后的问题，使其自包含：\n"
    "- 保留实体与时间范围；\n"
    "- 补齐指代（如‘它/这/上述’）；\n"
    "- 不要编造信息；\n"
    "- 只输出改写后的问题文本。"
)


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    try:
        if isinstance(content, str):
            return content
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
        return str(getattr(content, "content", content)).strip()
    except Exception:
        return str(content)


def _messages_to_text(messages: List[Any], max_chars: int = 3000) -> str:
    lines: List[str] = []
    # 仅取最近若干条，避免过长
    tail = messages[-8:] if len(messages) > 8 else messages
    for m in tail:
        role = None
        content = None
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            role = getattr(m, "type", None) or getattr(m, "role", None)
            content = getattr(m, "content", None)
        if not isinstance(content, str):
            content = _extract_text_content(content)
        role = role or "message"
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def build_short_term_memory_graph(agent: Any):
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.checkpoint.memory import InMemorySaver

    cfg = get_config()
    try:
        rewrite_model = init_model_with_config(cfg.response_model_name, cfg.response_model_temperature)
    except Exception:
        rewrite_model = None

    def call_agent(state: MessagesState) -> Dict[str, Any]:
        msgs = state.get("messages", [])
        if not msgs:
            return {"messages": []}
        # 构造改写输入
        conv_text = _messages_to_text(msgs)
        question_text = None
        if rewrite_model is not None:
            try:
                prompt = [
                    {"role": "system", "content": _REWRITE_PROMPT},
                    {"role": "user", "content": conv_text},
                ]
                resp = rewrite_model.invoke(prompt)
                question_text = _extract_text_content(getattr(resp, "content", resp)) or None
            except Exception:
                question_text = None
        # 回退到最后一条用户消息
        if not question_text:
            last = msgs[-1]
            if isinstance(last, dict):
                question_text = str(last.get("content", ""))
            else:
                question_text = str(getattr(last, "content", last))

        ans = agent.run(question_text)
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": ans.text,
                    "meta": {
                        "citations": ans.citations,
                        "confidence": ans.confidence,
                        "extra": ans.meta,
                    },
                }
            ]
        }


def rewrite_question(messages: List[Any]) -> str:
    """Rewrite the latest user question using prior conversation for self-contained query.

    Falls back to the last user content when model is unavailable or fails.
    """
    cfg = get_config()
    try:
        rewrite_model = init_model_with_config(cfg.response_model_name, cfg.response_model_temperature)
    except Exception:
        rewrite_model = None

    conv_text = _messages_to_text(messages)
    question_text: str | None = None
    if rewrite_model is not None:
        try:
            prompt = [
                {"role": "system", "content": _REWRITE_PROMPT},
                {"role": "user", "content": conv_text},
            ]
            resp = rewrite_model.invoke(prompt)
            question_text = _extract_text_content(getattr(resp, "content", resp)) or None
        except Exception:
            question_text = None
    if not question_text:
        # Fallback to last user message content
        for m in reversed(messages):
            role = None
            content = None
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
            else:
                role = getattr(m, "role", None)
                content = getattr(m, "content", None)
            if role == "user":
                question_text = str(content or "")
                break
        if not question_text:
            question_text = ""
    return question_text

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_agent)
    builder.add_edge(START, "agent")
    mem = InMemorySaver()
    graph = builder.compile(checkpointer=mem)
    return graph