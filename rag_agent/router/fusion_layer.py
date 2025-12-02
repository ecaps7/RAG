from __future__ import annotations

from typing import Dict, Iterable, List

from ..common.config import TOP_K, get_weights, get_config
from ..common.logging import get_logger
from ..common.types import ContextChunk, FusionResult, Intent
from ..common.utils import tokenize_zh


def _score_chunk(ch: ContextChunk, w_sim: float, w_rel: float, w_rec: float) -> float:
    return w_sim * ch.similarity + w_rel * ch.reliability + w_rec * ch.recency


def _dedup(chunks: Iterable[ContextChunk]) -> List[ContextChunk]:
    seen: set[str] = set()
    uniq: List[ContextChunk] = []
    for ch in chunks:
        key = (ch.source_type, ch.source_id, ch.citation or ch.title or "")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ch)
    return uniq


def _normalize_list(values: List[float], method: str = "minmax", eps: float = 1e-8) -> List[float]:
    if not values:
        return []
    if method == "zscore":
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
        std = (var ** 0.5)
        if std < eps:
            # 近乎常量：返回 0.5 常量，避免不稳定
            return [0.5 for _ in values]
        # 标准化并裁剪到 [-3, 3]，再线性映射到 [0, 1]
        out = []
        for v in values:
            z = (v - mean) / (std + eps)
            z = max(-3.0, min(3.0, z))
            out.append((z + 3.0) / 6.0)
        return out
    # 默认 min-max
    vmin = min(values)
    vmax = max(values)
    rng = vmax - vmin
    if rng < eps:
        # 常量分布：返回 0.5 常量，避免 0/1 极端化
        return [0.5 for _ in values]
    return [(v - vmin) / (rng + eps) for v in values]


def _chunk_similarity(a: ContextChunk, b: ContextChunk, stopwords: set[str]) -> float:
    """轻量片段相似度：中文分词后去停用词的 Jaccard 相似度。

    不依赖嵌入，适合融合层的多样化惩罚。
    """
    ta = set(tokenize_zh((a.content or "").lower())) - set(stopwords)
    tb = set(tokenize_zh((b.content or "").lower())) - set(stopwords)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / max(len(union), 1)


def _mmr_select(candidates: List[ContextChunk], scores: Dict[str, float], k: int, alpha: float, stopwords: set[str]) -> List[ContextChunk]:
    """在候选集中应用 MMR 选择 Top-K。

    目标：最大化相关性，同时最小化与已选集合的相似度。
    score(i) = alpha * relevance(i) - (1-alpha) * max_{j∈S} sim(i, j)
    """
    if not candidates:
        return []
    selected: List[ContextChunk] = []
    pool: List[ContextChunk] = list(candidates)

    while pool and len(selected) < k:
        if not selected:
            # 首个直接取最高分，确保强相关不被惩罚
            best = max(pool, key=lambda ch: scores.get(ch.id, 0.0))
            selected.append(best)
            pool.remove(best)
            continue
        best_item = None
        best_val = float("-inf")
        for ch in pool:
            rel = scores.get(ch.id, 0.0)
            div_penalty = 0.0
            try:
                div_penalty = max(_chunk_similarity(ch, s, stopwords) for s in selected)
            except Exception:
                div_penalty = 0.0
            val = alpha * rel - (1.0 - alpha) * div_penalty
            if val > best_val:
                best_val = val
                best_item = ch
        if best_item is None:
            break
        selected.append(best_item)
        pool.remove(best_item)
    return selected


class FusionLayer:
    def __init__(self, trace_id: str | None = None):
        self.logger = get_logger(self.__class__.__name__, trace_id)

    def aggregate(
        self,
        local: List[ContextChunk],
        web: List[ContextChunk],
        intent: Intent,
        k: int | None = None,
    ) -> FusionResult:
        weights = get_weights(intent)
        w_sim, w_rel, w_rec = weights["w_sim"], weights["w_rel"], weights["w_rec"]
        cfg = get_config()

        merged = _dedup([*local, *web])
        if not merged:
            self.logger.info("Fusion selected 0 chunks (empty inputs)")
            return FusionResult(selected_chunks=[], scores={})

        # ===== 归一化：对当前批次的 similarity/recency/reliability 分布做稳健归一化 =====
        scores: Dict[str, float] = {}
        if getattr(cfg, "fusion_use_normalization", True):
            method = str(getattr(cfg, "fusion_norm_method", "minmax")).lower()
            eps = float(getattr(cfg, "fusion_norm_eps", 1e-8))
            sim_vals = [ch.similarity for ch in merged]
            rec_vals = [ch.recency for ch in merged]
            rel_vals = [ch.reliability for ch in merged]

            sim_n = _normalize_list(sim_vals, method=method, eps=eps)
            rec_n = _normalize_list(rec_vals, method=method, eps=eps)
            rel_n = _normalize_list(rel_vals, method=method, eps=eps)

            # 线性加权融合分数（基于归一化后的特征）
            for i, ch in enumerate(merged):
                s = w_sim * sim_n[i] + w_rel * rel_n[i] + w_rec * rec_n[i]
                scores[ch.id] = float(s)
        else:
            # 保持原有行为：直接使用原始特征
            for ch in merged:
                s = _score_chunk(ch, w_sim, w_rel, w_rec)
                scores[ch.id] = float(s)

        # 先按融合分数排序，取 Top-N 作为候选
        merged.sort(key=lambda c: scores.get(c.id, 0.0), reverse=True)

        top_k = int(k) if isinstance(k, int) and k and k > 0 else int(TOP_K["fusion"])
        use_mmr = bool(getattr(cfg, "fusion_use_mmr", True))
        if use_mmr:
            mult = float(getattr(cfg, "fusion_mmr_fetch_multiplier", 2.5))
            alpha = float(getattr(cfg, "fusion_mmr_alpha", 0.35))
            top_n = max(top_k, int(max(1.0, mult) * top_k))
            candidates = merged[:top_n]
            selected = _mmr_select(candidates, scores, top_k, alpha, cfg.zh_stopwords)
        else:
            selected = merged[:top_k]

        avg_conf = sum(scores.get(ch.id, 0.0) for ch in selected) / max(len(selected), 1)
        self.logger.info(
            "Fusion selected %d chunks (avg=%.2f, mmr=%s, norm=%s)",
            len(selected), avg_conf, str(use_mmr), str(getattr(cfg, "fusion_use_normalization", True))
        )
        return FusionResult(selected_chunks=selected, scores=scores)