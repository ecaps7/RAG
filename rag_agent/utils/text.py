"""Text processing utilities for Chinese text."""

from __future__ import annotations

import re
from typing import List, Set

import jieba


# Synonym mapping for domain-specific terms (used for text normalization)
SYNONYM_MAP: dict[str, List[str]] = {
    "总资产": ["资产总额", "资产总计", "total assets"],
    "营业收入": ["营业总收入", "营收", "revenue"],
    "净利润": ["净收益", "净收入", "net income"],
    "净息差": ["净利差", "利差", "nim", "net interest margin"],
    "净利息收入": ["净息收入", "net interest income"],
    "资本充足率": ["car", "capital adequacy ratio"],
}


# Date expression expansion mapping
# Maps common date expressions to their equivalent forms found in documents
DATE_EXPANSION_MAP: dict[str, List[str]] = {
    # Month end expressions  
    "1月末": ["1月31日"],
    "2月末": ["2月28日", "2月29日"],
    "3月末": ["3月31日"],
    "4月末": ["4月30日"],
    "5月末": ["5月31日"],
    "6月末": ["6月30日"],
    "7月末": ["7月31日"],
    "8月末": ["8月31日"],
    "9月末": ["9月30日"],
    "10月末": ["10月31日"],
    "11月末": ["11月30日"],
    "12月末": ["12月31日"],
    # Quarter end
    "一季度末": ["3月31日"],
    "二季度末": ["6月30日"],
    "三季度末": ["9月30日"],
    "四季度末": ["12月31日"],
    # Year expressions
    "年末": ["12月31日"],
    "年初": ["1月1日"],
}


def expand_date_expressions(query: str) -> str:
    """Expand date expressions in query to include equivalent forms.
    
    For example: "9月末" -> "9月末 9月30日"
    
    Args:
        query: The original query
        
    Returns:
        Query with expanded date expressions
    """
    expanded = query
    for expr, alternatives in DATE_EXPANSION_MAP.items():
        if expr in query:
            additions = [alt for alt in alternatives if alt not in query]
            if additions:
                expanded = expanded + " " + " ".join(additions)
    return expanded


def tokenize_zh(text: str) -> List[str]:
    """Tokenize Chinese text using jieba."""
    return list(jieba.cut((text or "").lower()))


def normalize_terms(text: str) -> str:
    """Normalize synonyms to canonical terms."""
    t = (text or "").lower()
    for canon, variants in SYNONYM_MAP.items():
        for v in variants:
            if v and v in t:
                t = t.replace(v, canon)
    return t


def normalize_numbers(text: str) -> str:
    """Normalize numeric expressions (percentages and basis points).
    
    - Converts "1.2%" -> "1.2%(12bp)"
    - Converts "35bp" -> "35bp(0.35%)"
    - Converts "基点" to "bp"
    """
    t = (text or "")

    # Normalize Chinese "基点" to "bp"
    t = t.replace("基点", "bp")

    # Add bp tokens for percentages
    def pct_to_bp(m: re.Match) -> str:
        val = m.group(1)
        try:
            v = float(val)
            bp = int(round(v * 100))  # 1% -> 100bp; 0.35% -> 35bp
            return f"{val}%({bp}bp)"
        except Exception:
            return m.group(0)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", pct_to_bp, t)

    # Add percentage tokens for bp values
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
    """Compute Chinese word overlap ratio (with synonym normalization).
    
    Args:
        query_text: The query text
        doc_content: The document content
        stopwords: Set of stopwords to exclude
        
    Returns:
        Overlap ratio between 0 and 1
    """
    q_norm = normalize_numbers(normalize_terms(query_text))
    d_norm = normalize_numbers(normalize_terms(doc_content))
    q_words = set(tokenize_zh(q_norm)) - set(stopwords)
    d_words = set(tokenize_zh(d_norm)) - set(stopwords)
    if not q_words:
        return 0.0
    overlap = q_words & d_words
    return len(overlap) / len(q_words)
