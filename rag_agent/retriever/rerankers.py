from __future__ import annotations

from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore


def cross_encoder_rerank(query: str, pairs: List[Tuple[str, object]], model_name: str) -> List[Tuple[float, str, object]]:
    """对 (doc_text, payload) 对进行交叉编码器重排。

    pairs: 列表[(doc_text, payload)]，payload 可以是原始 Document 或 ContextChunk。
    返回：[(score, doc_text, payload)]，按分数降序。
    当库或模型不可用时，返回空列表以便调用方回退。
    """
    if CrossEncoder is None:
        return []
    try:
        model = CrossEncoder(model_name)
        inputs = [(query, text) for (text, _payload) in pairs]
        scores = model.predict(inputs)
        ranked = sorted(zip(list(scores), [t for (t, p) in pairs], [p for (t, p) in pairs]), key=lambda x: x[0], reverse=True)
        return ranked
    except Exception:
        return []