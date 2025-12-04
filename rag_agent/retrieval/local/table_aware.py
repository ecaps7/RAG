"""Table-aware retrieval utilities."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from ...core.types import ContextChunk


# Keywords indicating table-related queries
TABLE_QUERY_KEYWORDS: Set[str] = {
    # Numeric queries
    "多少", "是多少", "有多少", "数额", "数值", "金额",
    "余额", "总额", "合计", "净值", "本金",
    # Ratios and percentages
    "比例", "比率", "占比", "百分比", "百分点",
    "充足率", "覆盖率", "不良率", "拨备率", "杠杆率",
    "净息差", "净利差", "成本收入比", "迁徙率",
    # Comparisons
    "同比", "环比", "较上年", "较上季", "较上月",
    "增长", "下降", "变化", "变动", "增幅", "降幅", "增减",
    # Time points
    "截至", "期末", "期初", "年末", "季末", "月末",
    # Specific values
    "总资产", "总负债", "净利润", "营业收入", "净收入",
    "存款", "贷款", "资本", "收益率", "回报率",
    # Shareholder and equity related
    "股东", "持股", "股份", "股权", "前十名", "前三名", "最大",
}

# Source document patterns for filtering
SOURCE_PATTERNS: Dict[str, List[str]] = {
    "招商银行": ["CMB", "招商银行", "招行"],
    "中信银行": ["CITIC", "中信银行", "中信"],
    "工商银行": ["ICBC", "工商银行", "工行"],
    "建设银行": ["CCB", "建设银行", "建行"],
    "农业银行": ["ABC", "农业银行", "农行"],
    "中国银行": ["BOC", "中国银行", "中行"],
    "交通银行": ["BOCOM", "交通银行", "交行"],
}

# Keywords for definition/explanation queries (less table-focused)
DEFINITION_KEYWORDS: Set[str] = {
    "什么是", "如何定义", "定义", "含义", "口径",
    "计算方法", "计算公式", "怎么算", "如何计算",
    "解释", "说明", "意思",
}

# Time-related patterns
TIME_PATTERNS = [
    r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
    r"(\d{4})\s*年\s*(第?[一二三四1-4]季度|[1-4]Q|Q[1-4])",
    r"(\d{4})\s*年\s*(\d{1,2})\s*月",
    r"(\d{4})\s*年?(上半年|下半年|全年|年度|年末)",
    r"截至.*?(\d{4})[年/\-]",
]

# Report type patterns for strict filtering
# Maps keywords in query to filename patterns
REPORT_TYPE_PATTERNS: Dict[str, List[str]] = {
    # Half-year report (半年报)
    "半年报": ["-h1", "半年度", "半年报", "中期报告"],
    "半年度报告": ["-h1", "半年度", "半年报", "中期报告"],
    "中期报告": ["-h1", "半年度", "半年报", "中期报告"],
    "上半年": ["-h1", "半年度", "半年报", "中期报告"],  # Only if explicit context
    # Q1 report (一季报)
    "一季报": ["-q1", "一季度", "第一季度", "Q1"],
    "一季度报告": ["-q1", "一季度", "第一季度", "Q1"],
    "第一季度报告": ["-q1", "一季度", "第一季度", "Q1"],
    "第一季度": ["-q1", "一季度", "第一季度", "Q1"],
    # Q2 report - typically part of H1, but might be separate
    "二季度报告": ["-q2", "二季度", "第二季度", "Q2"],
    "第二季度报告": ["-q2", "二季度", "第二季度", "Q2"],
    # Q3 report (三季报)
    "三季报": ["-q3", "三季度", "第三季度", "Q3", "前三季度"],
    "三季度报告": ["-q3", "三季度", "第三季度", "Q3"],
    "第三季度报告": ["-q3", "三季度", "第三季度", "Q3"],
    "前三季度报告": ["-q3", "三季度", "第三季度", "Q3"],
    # Annual report (年报)
    "年报": ["-annual", "年度报告", "年报", "-ar"],
    "年度报告": ["-annual", "年度报告", "年报", "-ar"],
    "全年报告": ["-annual", "年度报告", "年报", "-ar"],
}


class TableQueryDetector:
    """Utility class for detecting table-related queries."""
    
    def __init__(
        self,
        table_keywords: Set[str] | None = None,
        definition_keywords: Set[str] | None = None,
    ):
        self.table_keywords = table_keywords or TABLE_QUERY_KEYWORDS
        self.definition_keywords = definition_keywords or DEFINITION_KEYWORDS
    
    def is_table_query(self, query: str) -> bool:
        """Determine if query is likely seeking table/numeric data."""
        return is_table_query(query)
    
    def extract_time(self, query: str) -> str:
        """Extract time point from query."""
        return extract_time_from_query(query)
    
    def extract_source_filter(self, query: str) -> List[str]:
        """Extract source document filter patterns from query."""
        return extract_source_filter(query)


class TableAwareScorer:
    """Scorer that boosts table chunks for table-related queries."""
    
    def __init__(self, table_keywords: Set[str] | None = None):
        self.table_keywords = table_keywords or TABLE_QUERY_KEYWORDS
        self.detector = TableQueryDetector()
    
    def score(self, chunk: ContextChunk, query: str) -> float:
        """Calculate table relevance boost score."""
        return score_table_relevance(chunk, query, self.table_keywords)


def is_table_query(query: str) -> bool:
    """Determine if query is likely seeking table/numeric data.
    
    Args:
        query: The user's query
        
    Returns:
        True if query appears to need table data
    """
    query_lower = query.lower()
    
    # Check for definition keywords first (strong signal for non-table query)
    def_score = sum(1 for kw in DEFINITION_KEYWORDS if kw in query_lower)
    
    # If definition keywords present, likely not a table query
    if def_score >= 1:
        return False
    
    # Check for table keywords
    table_score = sum(1 for kw in TABLE_QUERY_KEYWORDS if kw in query_lower)
    
    return table_score >= 1


def extract_time_from_query(query: str) -> str:
    """Extract time point from query.
    
    Args:
        query: The user's query
        
    Returns:
        Extracted time string or empty string
    """
    for pattern in TIME_PATTERNS:
        match = re.search(pattern, query)
        if match:
            return match.group(0)
    return ""


def is_comparison_query(query: str) -> bool:
    """Check if query is comparing multiple sources/entities.
    
    Args:
        query: The user's query
        
    Returns:
        True if query compares multiple entities
    """
    matched_count = 0
    for entity, aliases in SOURCE_PATTERNS.items():
        if entity in query:
            matched_count += 1
            continue
        for alias in aliases:
            if alias in query:
                matched_count += 1
                break
    
    # Also check for comparison keywords
    comparison_keywords = ["和", "与", "还是", "对比", "比较", "谁", "哪个", "哪家"]
    has_comparison_kw = any(kw in query for kw in comparison_keywords)
    
    return matched_count > 1 or (matched_count >= 1 and has_comparison_kw)


def extract_source_filter(query: str) -> List[str]:
    """Extract source document filter patterns from query.
    
    Args:
        query: The user's query
        
    Returns:
        List of source patterns to filter by (e.g., ["CMB", "招商银行"])
        Returns empty list if multiple sources are mentioned (comparison query)
    """
    matched_entities = []
    
    for entity, aliases in SOURCE_PATTERNS.items():
        entity_matched = False
        if entity in query:
            entity_matched = True
        else:
            for alias in aliases:
                if alias in query:
                    entity_matched = True
                    break
        if entity_matched:
            matched_entities.append(entity)
    
    # If multiple sources are mentioned, it's likely a comparison query
    # Don't apply source filter to avoid excluding relevant data
    if len(matched_entities) > 1:
        return []  # No filter for comparison queries
    
    # Single source - return its aliases for filtering
    if len(matched_entities) == 1:
        entity = matched_entities[0]
        return list(SOURCE_PATTERNS.get(entity, []))
    
    return []


def extract_report_type_filter(query: str) -> List[str]:
    """Extract report type filter patterns from query.
    
    When user explicitly mentions a report type (e.g., "半年报", "三季报"),
    this function returns patterns to filter documents strictly by that type.
    
    Args:
        query: The user's query
        
    Returns:
        List of report type patterns to filter by (e.g., ["-h1", "半年度"])
        Returns empty list if no specific report type is mentioned.
    """
    # Check for explicit report type mentions
    for keyword, patterns in REPORT_TYPE_PATTERNS.items():
        if keyword in query:
            return patterns
    
    return []


def filter_by_report_type(docs, query: str):
    """Filter documents to only include those matching the report type in query.
    
    Args:
        docs: List of documents with metadata
        query: The user's query
        
    Returns:
        Filtered list of documents matching the report type, or original list
        if no report type filter is specified.
    """
    report_patterns = extract_report_type_filter(query)
    if not report_patterns:
        return docs
    
    filtered = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        source = str(meta.get("source", "")).lower()
        
        # Check if source matches any of the report type patterns
        matched = any(p.lower() in source for p in report_patterns)
        if matched:
            filtered.append(d)
    
    # If filtering removes all docs, return original (fallback)
    if not filtered:
        return docs
    
    return filtered


def score_table_relevance(
    chunk: ContextChunk,
    query: str,
    table_keywords: Set[str] | None = None,
) -> float:
    """Calculate additional relevance score for table chunks.
    
    Args:
        chunk: The context chunk to score
        query: The user's query
        table_keywords: Optional custom keywords set
        
    Returns:
        Relevance boost score (0.0 to 0.5)
    """
    if table_keywords is None:
        table_keywords = TABLE_QUERY_KEYWORDS
    
    boost = 0.0
    metadata = chunk.metadata or {}
    is_table = metadata.get("doctype") == "table"
    query_is_table = is_table_query(query)
    
    # Source matching boost (highest priority)
    source_filters = extract_source_filter(query)
    if source_filters:
        source = metadata.get("source", "")
        source_matched = any(f.upper() in source.upper() for f in source_filters)
        if source_matched:
            boost += 0.2  # Strong boost for matching source
        else:
            boost -= 0.15  # Penalty for non-matching source when filter is specified
    
    # Base boost for matching query type
    if query_is_table and is_table:
        boost += 0.15
    
    # Time point matching
    query_time = extract_time_from_query(query)
    chunk_time = metadata.get("time_point", "")
    
    if query_time and chunk_time:
        # Exact match
        if query_time in chunk_time or chunk_time in query_time:
            boost += 0.1
        # Year match
        elif re.search(r"\d{4}", query_time) and re.search(r"\d{4}", chunk_time):
            query_year = re.search(r"\d{4}", query_time).group()
            chunk_year = re.search(r"\d{4}", chunk_time).group()
            if query_year == chunk_year:
                boost += 0.05
    
    # Critical keyword matching in content (股东相关查询最重要的匹配)
    content = (chunk.content or "").lower()
    shareholder_keywords = ["股东", "持股", "股东情况", "前三名", "前十名", "股份"]
    query_lower = query.lower()
    for kw in shareholder_keywords:
        if kw in query_lower and kw in content:
            boost += 0.15  # Strong boost for keyword match in both query and content
            break
    
    # Definition query keyword matching (定义类查询关键词匹配)
    definition_patterns = ["包括", "包含", "定义", "是指", "指的是", "分为", "分别为"]
    definition_query_hints = ["包含", "包括", "哪些", "什么", "定义", "构成", "组成"]
    query_is_definition = any(h in query_lower for h in definition_query_hints)
    content_has_definition = any(p in content for p in definition_patterns)
    if query_is_definition and content_has_definition:
        # Check for term overlap (e.g., FPA in both query and content)
        query_upper = query.upper()
        content_upper = (chunk.content or "").upper()
        if "FPA" in query_upper and "FPA" in content_upper:
            boost += 0.2  # Strong boost for definition match
        elif "融资" in query and "融资" in content:
            boost += 0.15
    
    # Keyword overlap in table columns
    if is_table:
        columns = metadata.get("columns", [])
        keywords = metadata.get("keywords", [])
        # Ensure columns and keywords are lists
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(keywords, str):
            keywords = [keywords]
        all_terms = set(list(columns) + list(keywords))
        
        query_terms = set(query.lower().split())
        overlap = len(all_terms & query_terms)
        if overlap > 0:
            boost += min(0.05 * overlap, 0.1)
    
    return max(min(boost, 0.6), -0.15)  # Cap between -0.15 and 0.6 (increased max)


def reorder_with_table_preference(
    chunks: List[ContextChunk],
    query: str,
    table_ratio: float = 0.7,
) -> List[ContextChunk]:
    """Reorder chunks to prioritize tables for table queries.
    
    Args:
        chunks: List of retrieved chunks
        query: The user's query
        table_ratio: Target ratio of table chunks for table queries
        
    Returns:
        Reordered list of chunks
    """
    if not chunks or not is_table_query(query):
        return chunks
    
    # Separate table and text chunks
    table_chunks = []
    text_chunks = []
    
    for chunk in chunks:
        if (chunk.metadata or {}).get("doctype") == "table":
            table_chunks.append(chunk)
        else:
            text_chunks.append(chunk)
    
    # Calculate target counts
    total = len(chunks)
    target_tables = int(total * table_ratio)
    target_text = total - target_tables
    
    # Reorder: tables first (up to ratio), then text
    result = []
    result.extend(table_chunks[:target_tables])
    result.extend(text_chunks[:target_text])
    
    # Add remaining chunks
    result.extend(table_chunks[target_tables:])
    result.extend(text_chunks[target_text:])
    
    return result[:total]


def get_table_context_summary(chunks: List[ContextChunk]) -> str:
    """Generate a summary of table data in chunks for LLM context.
    
    Args:
        chunks: List of context chunks
        
    Returns:
        Summary string
    """
    table_summaries = []
    
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata or {}
        if metadata.get("doctype") != "table":
            continue
        
        parts = [f"表格{i+1}"]
        
        if metadata.get("table_name"):
            parts.append(f"标题: {metadata['table_name']}")
        
        if metadata.get("time_point"):
            parts.append(f"时点: {metadata['time_point']}")
        
        if metadata.get("unit"):
            parts.append(f"单位: {metadata['unit']}")
        
        columns = metadata.get("columns", [])
        if columns:
            cols_str = ", ".join(c for c in columns[:5] if c and not c.startswith("列"))
            if cols_str:
                parts.append(f"字段: {cols_str}")
        
        table_summaries.append(" | ".join(parts))
    
    if table_summaries:
        return "可用表格数据:\n" + "\n".join(table_summaries)
    return ""
