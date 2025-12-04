"""Table enhancement and semantic enrichment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .pdf_processor import ExtractedTable
from .chunker import extract_entity_from_source  # Import entity extraction


@dataclass
class EnhancedTable:
    """Table with semantic enrichment."""
    original: ExtractedTable
    enhanced_content: str  # Full enhanced content for indexing
    semantic_description: str  # Natural language description
    keywords: List[str]  # Extracted keywords for BM25
    metadata: Dict[str, Any]


class TableEnhancer:
    """Enhances extracted tables with semantic information.
    
    Features:
    - Generate natural language descriptions
    - Extract keywords for better retrieval
    - Enrich metadata
    - Create searchable representations
    - Inject source entity name for better retrieval
    """
    
    # Financial terms for keyword extraction
    FINANCIAL_TERMS = {
        # Assets
        "资产", "总资产", "流动资产", "非流动资产", "固定资产", "无形资产",
        "现金", "存款", "贷款", "投资", "应收", "预付",
        # Liabilities
        "负债", "总负债", "流动负债", "非流动负债", "应付", "预收",
        "借款", "债券", "存款",
        # Equity
        "股东权益", "所有者权益", "净资产", "资本", "公积", "留存收益",
        "未分配利润",
        # Income
        "收入", "营业收入", "利息收入", "手续费", "佣金", "投资收益",
        # Expenses
        "支出", "成本", "费用", "管理费", "销售费", "财务费",
        "减值损失", "税金",
        # Profit
        "利润", "净利润", "营业利润", "利润总额", "毛利", "息差",
        # Ratios
        "比率", "比例", "率", "占比", "充足率", "覆盖率", "不良率",
        "净息差", "净利差", "成本收入比", "资本充足率", "杠杆率",
        # Periods
        "季度", "半年", "年度", "期末", "期初", "同比", "环比",
        "较上年", "增长", "下降", "变动",
        # Entities
        "本行", "本公司", "本集团", "母公司", "合并", "单体",
    }
    
    # Common column patterns
    COLUMN_PATTERNS = {
        "金额": r"金额|余额|本金|价值",
        "比例": r"比例|占比|百分比|%",
        "增幅": r"增幅|增长|变动|增减|同比|环比",
        "日期": r"\d{4}.*?[月日季]|\d{4}/\d{1,2}",
    }
    
    def __init__(
        self,
        include_raw_data: bool = False,
        max_description_length: int = 500,
    ):
        self.include_raw_data = include_raw_data
        self.max_description_length = max_description_length
    
    def enhance(self, table: ExtractedTable, source_name: str = "") -> EnhancedTable:
        """Enhance a single table with semantic information.
        
        Args:
            table: The extracted table to enhance
            source_name: Source document name
            
        Returns:
            EnhancedTable with enriched content
        """
        # Generate semantic description
        description = self._generate_description(table)
        
        # Extract keywords
        keywords = self._extract_keywords(table)
        
        # Build enhanced content
        enhanced_content = self._build_enhanced_content(
            table, description, source_name
        )
        
        # Build enriched metadata
        metadata = self._build_metadata(table, keywords, source_name)
        
        return EnhancedTable(
            original=table,
            enhanced_content=enhanced_content,
            semantic_description=description,
            keywords=keywords,
            metadata=metadata,
        )
    
    def _generate_description(self, table: ExtractedTable) -> str:
        """Generate natural language description of table."""
        parts = []
        
        # Basic structure info
        col_count = len(table.columns)
        parts.append(f"该表格包含{table.row_count}行{col_count}列数据")
        
        # Column information
        real_columns = [c for c in table.columns if not c.startswith("列")]
        if real_columns:
            cols_preview = ", ".join(real_columns[:5])
            if len(real_columns) > 5:
                cols_preview += f" 等{len(real_columns)}个字段"
            parts.append(f"主要字段包括: {cols_preview}")
        
        # Unit information
        if table.unit:
            parts.append(f"数据单位为{table.unit}")
        
        # Time point
        if table.time_point:
            parts.append(f"数据时点为{table.time_point}")
        
        # Title context
        if table.title:
            parts.append(f"表格标题: {table.title}")
        
        # Detect data patterns
        patterns = self._detect_data_patterns(table)
        if patterns:
            parts.append(f"数据类型: {', '.join(patterns)}")
        
        description = "。".join(parts)
        
        # Truncate if too long
        if len(description) > self.max_description_length:
            description = description[:self.max_description_length] + "..."
        
        return description
    
    def _detect_data_patterns(self, table: ExtractedTable) -> List[str]:
        """Detect what types of data the table contains."""
        patterns = []
        
        # Flatten all text in table
        all_text = " ".join(table.columns)
        for row in table.raw_data:
            all_text += " " + " ".join(str(c) for c in row)
        
        # Check for patterns
        if re.search(r"同比|环比|增[长减]|变动", all_text):
            patterns.append("同比/环比变动")
        if re.search(r"期初|期末|余额", all_text):
            patterns.append("期初期末余额")
        if re.search(r"资产|负债|权益", all_text):
            patterns.append("资产负债数据")
        if re.search(r"收入|支出|利润", all_text):
            patterns.append("损益数据")
        if re.search(r"充足率|覆盖率|比率", all_text):
            patterns.append("监管指标")
        if re.search(r"贷款|存款|不良", all_text):
            patterns.append("信贷资产数据")
        
        return patterns
    
    def _extract_keywords(self, table: ExtractedTable) -> List[str]:
        """Extract keywords from table for BM25 indexing."""
        keywords = set()
        
        # From columns
        for col in table.columns:
            keywords.add(col)
            # Also add individual terms
            for term in self.FINANCIAL_TERMS:
                if term in col:
                    keywords.add(term)
        
        # From title
        if table.title:
            for term in self.FINANCIAL_TERMS:
                if term in table.title:
                    keywords.add(term)
        
        # From context
        context = f"{table.context_before} {table.context_after}"
        for term in self.FINANCIAL_TERMS:
            if term in context:
                keywords.add(term)
        
        # From data (first column often contains labels)
        if table.raw_data:
            for row in table.raw_data[:20]:  # Limit to first 20 rows
                if row and row[0]:
                    first_cell = str(row[0])
                    for term in self.FINANCIAL_TERMS:
                        if term in first_cell:
                            keywords.add(term)
                    # Add the label itself if it's text
                    if not first_cell.replace(",", "").replace(".", "").replace("-", "").isdigit():
                        keywords.add(first_cell)
        
        return list(keywords)
    
    def _build_enhanced_content(
        self,
        table: ExtractedTable,
        description: str,
        source_name: str,
    ) -> str:
        """Build enhanced content string for indexing.
        
        Injects source entity name (e.g., bank name) to improve BM25 retrieval.
        """
        parts = []
        
        # Inject source entity name at the beginning for BM25 retrieval
        entity_name = extract_entity_from_source(source_name)
        if entity_name:
            parts.append(f"【来源】{entity_name}")
        
        parts.append("[TABLE]")
        
        # Metadata section
        if table.title:
            parts.append(f"【表格标题】{table.title}")
        
        if table.unit:
            parts.append(f"【单位】{table.unit}")
        
        if table.time_point:
            parts.append(f"【时点】{table.time_point}")
        
        # Context section
        if table.context_before:
            context = table.context_before[:200]
            parts.append(f"【上下文】{context}")
        
        # Description
        parts.append(f"【数据描述】{description}")
        
        # Column summary
        real_columns = [c for c in table.columns if c]
        if real_columns:
            parts.append(f"【字段】{' | '.join(real_columns)}")
        
        # Main content (markdown table)
        parts.append("【数据内容】")
        parts.append(table.content)
        
        return "\n".join(parts)
    
    def _build_metadata(
        self,
        table: ExtractedTable,
        keywords: List[str],
        source_name: str,
    ) -> Dict[str, Any]:
        """Build enriched metadata dictionary."""
        # Extract entity name for metadata
        entity_name = extract_entity_from_source(source_name)
        
        metadata = {
            "source": source_name,
            "page": table.page,
            "table_index": table.table_index,
            "doctype": "table",
            "repr": "enhanced_markdown",
            "row_count": table.row_count,
            "col_count": len(table.columns),
            "columns": table.columns,
            "table_name": table.title,
            "unit": table.unit,
            "time_point": table.time_point,
            "keywords": keywords,
            "has_context": bool(table.context_before or table.context_after),
            "enhanced": True,
        }
        
        # Add entity name if extracted
        if entity_name:
            metadata["source_entity"] = entity_name
            # Also add entity to keywords for BM25
            if entity_name not in keywords:
                metadata["keywords"] = keywords + [entity_name]
        
        return metadata


def enhance_tables(
    tables: List[ExtractedTable],
    source_name: str = "",
) -> List[EnhancedTable]:
    """Convenience function to enhance multiple tables."""
    enhancer = TableEnhancer()
    return [enhancer.enhance(t, source_name) for t in tables]
