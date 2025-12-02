from __future__ import annotations

from typing import List, Set

import jieba


def tokenize_zh(text: str) -> List[str]:
    return list(jieba.cut((text or "").lower()))


_SYNONYM_MAP: dict[str, List[str]] = {
    "总资产": ["资产总额", "资产总计", "total assets"],
    "营业收入": ["营业总收入", "营收", "revenue"],
    "净利润": ["净收益", "净收入", "net income"],
    # 领域扩展
    "净息差": ["净利差", "利差", "nim", "net interest margin"],
    "净利息收入": ["净息收入", "net interest income"],
    "资本充足率": ["car", "capital adequacy ratio"],
}


def _normalize_terms(text: str) -> str:
    t = (text or "").lower()
    for canon, variants in _SYNONYM_MAP.items():
        for v in variants:
            if v and v in t:
                t = t.replace(v, canon)
    return t


def _normalize_numbers(text: str) -> str:
    """统一数字表达：将百分比与基点转换为统一令牌，便于匹配。

    规则：
    - 识别 e.g. "1.2%" -> "12bp"（近似：1% = 100bp）
    - 识别 e.g. "35bp" -> "0.35%"（双向提供令牌，增强召回）
    - 统一中文“基点”为 bp
    - 保留原文，但追加规范化令牌，避免丢信息
    """
    t = (text or "")

    # 统一中文“基点”
    t = t.replace("基点", "bp")

    # 百分比 -> bp 的附加令牌
    import re
    def pct_to_bp(m: re.Match) -> str:
        val = m.group(1)
        try:
            v = float(val)
            bp = int(round(v * 100))  # 1% -> 100bp；0.35% -> 35bp
            return f"{val}%({bp}bp)"
        except Exception:
            return m.group(0)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", pct_to_bp, t)

    # bp -> 百分比 的附加令牌
    def bp_to_pct(m: re.Match) -> str:
        val = m.group(1)
        try:
            v = float(val)
            pct = round(v / 100.0, 4)
            return f"{val}bp({pct}%)"
        except Exception:
            return m.group(0)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*bp", bp_to_pct, t, flags=re.IGNORECASE)

    return t


def compute_overlap_ratio(query_text: str, doc_content: str, stopwords: Set[str]) -> float:
    """计算中文词重叠率（去停用词 + 同义词规范化 + 数字标准化）。"""
    q_norm = _normalize_numbers(_normalize_terms(query_text))
    d_norm = _normalize_numbers(_normalize_terms(doc_content))
    q_words = set(tokenize_zh(q_norm)) - set(stopwords)
    d_words = set(tokenize_zh(d_norm)) - set(stopwords)
    if not q_words:
        return 0.0
    overlap = q_words & d_words
    return len(overlap) / len(q_words)