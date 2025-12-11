"""Table extraction and summarization for financial reports.

This module processes tables from Marker JSON output, implementing 
a "Summary + Source Code" dual-track approach:
1. Extract raw table code (HTML) for injection into LLM prompts during RAG generation
2. Generate natural language summaries for vector embedding retrieval

Input: Marker JSON format with block_type="Table"
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TableConfig:
    """Configuration for table processing."""
    
    # OpenAI-compatible API settings (configurable via environment or direct init)
    api_base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 1024
    
    def __post_init__(self):
        """Load defaults from environment if not provided."""
        if not self.api_base_url:
            self.api_base_url = os.getenv(
                "TABLE_LLM_BASE_URL",
                os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            )
        if not self.api_key:
            self.api_key = os.getenv(
                "TABLE_LLM_API_KEY",
                os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
            )
        if not self.model:
            self.model = os.getenv(
                "TABLE_LLM_MODEL",
                os.getenv("LLM_MODEL", "deepseek-v3.2")
            )
    
    @classmethod
    def from_env(cls) -> "TableConfig":
        """Create config from environment variables."""
        return cls()
    
    @classmethod
    def create(
        cls,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "TableConfig":
        """Create config with custom settings."""
        config = cls()
        if base_url:
            config.api_base_url = base_url
        if api_key:
            config.api_key = api_key
        if model:
            config.model = model
        if temperature is not None:
            config.temperature = temperature
        if max_tokens is not None:
            config.max_tokens = max_tokens
        return config


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TableContext:
    """Contextual information surrounding a table."""
    preceding_text: str = ""    # Text paragraph before the table
    following_text: str = ""    # Text paragraph after the table
    section_path: List[str] = field(default_factory=list)  # Section hierarchy


@dataclass
class DocumentContext:
    """Global document context for disambiguation."""
    company_name: str = ""      # Full company name (e.g., "招商银行股份有限公司")
    company_short: str = ""     # Short name (e.g., "招商银行")
    stock_code: str = ""        # Stock code (e.g., "600036.SH")
    report_period: str = ""     # Report period (e.g., "2025年第一季度")
    report_type: str = ""       # Report type: 季报/半年报/年报
    fiscal_year: str = ""       # Fiscal year (e.g., "2025")
    data_scope: str = ""        # Data scope: 本行/集团
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "company_name": self.company_name,
            "company_short": self.company_short,
            "stock_code": self.stock_code,
            "report_period": self.report_period,
            "report_type": self.report_type,
            "fiscal_year": self.fiscal_year,
            "data_scope": self.data_scope,
        }


# Company code to name mapping
COMPANY_MAPPING = {
    "CMB": ("招商银行股份有限公司", "招商银行", "600036.SH / 03968.HK"),
    "CITIC": ("中信银行股份有限公司", "中信银行", "601998.SH / 00998.HK"),
    "ICBC": ("中国工商银行股份有限公司", "工商银行", "601398.SH / 01398.HK"),
    "CCB": ("中国建设银行股份有限公司", "建设银行", "601939.SH / 00939.HK"),
    "ABC": ("中国农业银行股份有限公司", "农业银行", "601288.SH / 01288.HK"),
    "BOC": ("中国银行股份有限公司", "中国银行", "601988.SH / 03988.HK"),
    "PSBC": ("中国邮政储蓄银行股份有限公司", "邮储银行", "601658.SH / 01658.HK"),
    "BOCOM": ("交通银行股份有限公司", "交通银行", "601328.SH / 03328.HK"),
    "CEB": ("中国光大银行股份有限公司", "光大银行", "601818.SH / 06818.HK"),
    "CMBC": ("中国民生银行股份有限公司", "民生银行", "600016.SH / 01988.HK"),
    "CIB": ("兴业银行股份有限公司", "兴业银行", "601166.SH"),
    "SPDB": ("上海浦东发展银行股份有限公司", "浦发银行", "600000.SH"),
    "HXB": ("华夏银行股份有限公司", "华夏银行", "600015.SH"),
    "PAB": ("平安银行股份有限公司", "平安银行", "000001.SZ"),
}

# Report type mapping
REPORT_TYPE_MAPPING = {
    "q1": ("第一季度", "季报"),
    "q2": ("第二季度", "季报"),
    "q3": ("第三季度", "季报"),
    "q4": ("第四季度", "季报"),
    "h1": ("上半年", "半年报"),
    "h2": ("下半年", "半年报"),
    "annual": ("全年", "年报"),
    "year": ("全年", "年报"),
}


def parse_document_context(source_name: str) -> DocumentContext:
    """Parse document context from source filename.
    
    Expected filename patterns:
    - CMB-2025-q1 -> 招商银行 2025年第一季度 季报
    - CITIC-2025-h1 -> 中信银行 2025年上半年 半年报
    
    Args:
        source_name: Source document filename
        
    Returns:
        DocumentContext with parsed information
    """
    context = DocumentContext()
    
    if not source_name:
        return context
    
    # Remove file extension
    name = source_name.replace(".json", "").replace(".md", "").replace(".pdf", "")
    
    # Try to parse: COMPANY-YEAR-PERIOD pattern
    parts = name.upper().split("-")
    
    # Extract company info
    for code, (full_name, short_name, stock_code) in COMPANY_MAPPING.items():
        if code.upper() in parts or code.upper() in name.upper():
            context.company_name = full_name
            context.company_short = short_name
            context.stock_code = stock_code
            break
    
    # Extract year
    for part in parts:
        if part.isdigit() and len(part) == 4:
            context.fiscal_year = part
            break
    
    # Extract report period/type
    name_lower = name.lower()
    for period_code, (period_name, report_type) in REPORT_TYPE_MAPPING.items():
        if period_code in name_lower:
            if context.fiscal_year:
                context.report_period = f"{context.fiscal_year}年{period_name}"
            else:
                context.report_period = period_name
            context.report_type = report_type
            break
    
    # Default report period if year found but no period
    if context.fiscal_year and not context.report_period:
        context.report_period = f"{context.fiscal_year}年"
        context.report_type = "年报"
    
    # Default data scope (can be overridden by content analysis)
    context.data_scope = "集团"
    
    return context


@dataclass
class ExtractedTable:
    """A table extracted from Marker JSON."""
    id: str                           # Unique table identifier (e.g., TABLE_0)
    block_id: str                     # Original block ID from JSON
    raw_code: str                     # Original table HTML code
    page: int                         # Page number in source document
    context: TableContext             # Surrounding context
    source: str = ""                  # Source document name
    bbox: List[float] = field(default_factory=list)  # Bounding box


@dataclass  
class ProcessedTable:
    """A fully processed table with summary."""
    id: str = ""                      # Unique table identifier
    content: str = ""                 # LLM-generated summary
    page: int = 0                     # Page number (0-based internally)
    document_context: DocumentContext = field(default_factory=DocumentContext)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to simplified dictionary format."""
        return {
            "id": self.id,
            "summary": self.content,
            "page": self.page + 1,  # 改为 1-based
            "section": self.metadata.get("section_path", []),
            "raw_html": self.metadata.get("raw_code", ""),
            "context": {
                "before": self.metadata.get("preceding_text", ""),
                "after": self.metadata.get("following_text", "")
            },
            "bbox": self.metadata.get("bbox", [])
        }


# ============================================================================
# HTML Utilities
# ============================================================================

class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        
    def handle_data(self, data):
        text = data.strip()
        if text:
            self.text_parts.append(text)
            
    def get_text(self) -> str:
        return ' '.join(self.text_parts).strip()


def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML string."""
    if not html:
        return ""
    parser = HTMLTextExtractor()
    try:
        parser.feed(html)
    except:
        return html
    return parser.get_text()


def extract_header_text(html: str) -> str:
    """Extract header text from HTML (h1, h2, h3, etc.)."""
    match = re.search(r'<h[1-6][^>]*>(.*?)</h[1-6]>', html, re.DOTALL | re.IGNORECASE)
    if match:
        inner = match.group(1)
        inner = re.sub(r'<[^>]+>', ' ', inner)
        return inner.strip()
    return extract_text_from_html(html)


# ============================================================================
# JSON Parser - Extract Tables from Marker JSON
# ============================================================================

class MarkerJSONTableParser:
    """Parse Marker JSON output and extract tables with context."""
    
    # Block types for tables
    TABLE_BLOCK_TYPES = {'Table', 'TableOfContents'}
    
    # Block types for text (for context extraction)
    TEXT_BLOCK_TYPES = {'Text', 'SectionHeader', 'ListItem', 'Caption'}
    
    # Block types to skip
    SKIP_BLOCK_TYPES = {'PageHeader', 'PageFooter'}
    
    def __init__(self, source_name: str = ""):
        self.source_name = source_name
        self.blocks: List[Dict] = []  # All blocks in order
        self.section_headers: Dict[str, str] = {}  # id -> header text
        self._table_counter = 0
        
    def parse(self, json_data: dict) -> List[ExtractedTable]:
        """Parse JSON and extract tables with context.
        
        Args:
            json_data: Marker JSON data
            
        Returns:
            List of ExtractedTable objects
        """
        self.blocks = []
        self.section_headers = {}
        self._table_counter = 0
        
        # First pass: collect all blocks and section headers
        for page_block in json_data.get('children', []):
            if page_block.get('block_type') == 'Page':
                for child in page_block.get('children', []):
                    self._collect_block(child)
        
        # Second pass: extract tables with context
        tables = []
        for i, block in enumerate(self.blocks):
            if block.get('block_type') in self.TABLE_BLOCK_TYPES:
                table = self._extract_table(block, i)
                tables.append(table)
                
        return tables
    
    def _collect_block(self, block_data: dict):
        """Collect block and extract section header text."""
        block_type = block_data.get('block_type', '')
        block_id = block_data.get('id', '')
        
        if block_type in self.SKIP_BLOCK_TYPES:
            return
            
        # Store section header text for hierarchy resolution
        if block_type == 'SectionHeader':
            html = block_data.get('html', '')
            text = extract_header_text(html)
            self.section_headers[block_id] = text
            
        self.blocks.append(block_data)
    
    def _extract_table(self, block: dict, block_index: int) -> ExtractedTable:
        """Extract a single table with its context."""
        self._table_counter += 1
        table_id = f"TABLE_{self._table_counter - 1}"
        
        block_id = block.get('id', '')
        html = block.get('html', '')
        page = block.get('page', 0)
        bbox = block.get('bbox', [])
        section_hierarchy = block.get('section_hierarchy', {})
        
        # Resolve section path
        section_path = self._resolve_section_path(section_hierarchy)
        
        # Get surrounding text context
        preceding_text = self._get_preceding_text(block_index)
        following_text = self._get_following_text(block_index)
        
        context = TableContext(
            preceding_text=preceding_text,
            following_text=following_text,
            section_path=section_path,
        )
        
        return ExtractedTable(
            id=table_id,
            block_id=block_id,
            raw_code=html,
            page=page,
            context=context,
            source=self.source_name,
            bbox=bbox,
        )
    
    def _resolve_section_path(self, hierarchy: Dict[str, str]) -> List[str]:
        """Resolve section hierarchy to a list of header texts."""
        path = []
        for level in sorted(hierarchy.keys(), key=lambda x: int(x)):
            header_id = hierarchy[level]
            if header_id in self.section_headers:
                path.append(self.section_headers[header_id])
        return path
    
    def _get_preceding_text(self, block_index: int, max_blocks: int = 3) -> str:
        """Get text from blocks before the table."""
        texts = []
        for i in range(block_index - 1, max(0, block_index - max_blocks - 1), -1):
            block = self.blocks[i]
            if block.get('block_type') in self.TEXT_BLOCK_TYPES:
                html = block.get('html', '')
                text = extract_text_from_html(html)
                if text:
                    texts.insert(0, text)
        return '\n'.join(texts)
    
    def _get_following_text(self, block_index: int, max_blocks: int = 3) -> str:
        """Get text from blocks after the table."""
        texts = []
        for i in range(block_index + 1, min(len(self.blocks), block_index + max_blocks + 1)):
            block = self.blocks[i]
            if block.get('block_type') in self.TEXT_BLOCK_TYPES:
                html = block.get('html', '')
                text = extract_text_from_html(html)
                if text:
                    texts.append(text)
        return '\n'.join(texts)


# ============================================================================
# Table Summarization (LLM)
# ============================================================================

class TableSummarizer:
    """Generate natural language summaries for tables using LLM."""
    
    # Prompt template for table summarization
    SUMMARIZE_PROMPT = """你是一位专业的财务分析师。请分析以下财务报表/表格，用自然语言总结其关键内容。

要求：
1. 在摘要开头明确指出这是哪家公司、什么时期的数据
2. 用简洁的中文描述表格的主题和核心数据
3. 提取关键的增长数据、同比变化、环比变化
4. 指出任何异常值或显著变化
5. 总结字数控制在150-300字以内

## 文档全局信息
- 公司名称：{company_name}
- 报告期间：{report_period}
- 报告类型：{report_type}
- 数据口径：{data_scope}

## 表格上下文
- 章节路径：{section_path}

## 表格前的说明文字
{preceding_text}

## 表格内容 (HTML格式)
```html
{table_code}
```

## 表格后的注释/说明
{following_text}

请输出表格摘要（开头需明确公司和时期）："""

    def __init__(self, config: Optional[TableConfig] = None):
        """Initialize summarizer."""
        self.config = config or TableConfig.from_env()
        self._client: Optional[OpenAI] = None
    
    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client (lazy initialization)."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base_url,
            )
        return self._client
    
    def summarize(
        self,
        table: ExtractedTable,
        document_context: Optional[DocumentContext] = None,
    ) -> str:
        """Generate a summary for a table."""
        doc_ctx = document_context or parse_document_context(table.source)
        
        section_path_str = " > ".join(table.context.section_path) if table.context.section_path else "未知"
        
        prompt = self.SUMMARIZE_PROMPT.format(
            company_name=doc_ctx.company_name or "未知公司",
            report_period=doc_ctx.report_period or "未知期间",
            report_type=doc_ctx.report_type or "未知",
            data_scope=doc_ctx.data_scope or "未知",
            section_path=section_path_str,
            preceding_text=table.context.preceding_text or "（无）",
            table_code=table.raw_code,
            following_text=table.context.following_text or "（无）",
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是专业的财务报表分析助手，擅长解读财务数据并用简洁的语言总结关键信息。"
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            summary = response.choices[0].message.content or ""
            return summary.strip()
            
        except Exception as e:
            return self._fallback_summary(table, str(e))
    
    def _fallback_summary(self, table: ExtractedTable, error: str) -> str:
        """Generate fallback summary when LLM fails."""
        section_info = " > ".join(table.context.section_path) if table.context.section_path else ""
        return f"[表格摘要生成失败: {error}] 这是一张表格，位于第{table.page + 1}页。章节：{section_info}"


# ============================================================================
# Table Processor (Main Interface)
# ============================================================================

class TableProcessor:
    """Main interface for table extraction and processing from Marker JSON."""
    
    def __init__(
        self,
        config: Optional[TableConfig] = None,
        source_name: str = "",
        document_context: Optional[DocumentContext] = None,
    ):
        self.config = config or TableConfig.from_env()
        self.source_name = source_name
        self.document_context = document_context or parse_document_context(source_name)
        self.summarizer = TableSummarizer(config)
    
    def process_json(
        self,
        json_data: dict,
        generate_summaries: bool = True,
    ) -> List[ProcessedTable]:
        """Process Marker JSON to extract and summarize tables.
        
        Args:
            json_data: Marker JSON data
            generate_summaries: Whether to generate LLM summaries
            
        Returns:
            List of ProcessedTable objects
        """
        # Parse JSON and extract tables
        parser = MarkerJSONTableParser(self.source_name)
        tables = parser.parse(json_data)
        
        # Process each table
        processed = []
        for table in tables:
            processed_table = self._process_single_table(table, generate_summaries)
            processed.append(processed_table)
            print(f"Processed table {table.id} on page {table.page + 1}")
        
        return processed
    
    def _process_single_table(
        self, 
        table: ExtractedTable, 
        generate_summary: bool = True
    ) -> ProcessedTable:
        """Process a single extracted table."""
        # Generate summary if requested
        if generate_summary:
            summary = self.summarizer.summarize(table, self.document_context)
        else:
            summary = ""
        
        # Build metadata
        metadata = {
            "source": table.source,
            "page": table.page,
            "section_path": table.context.section_path,
            "raw_code": table.raw_code,
            "bbox": table.bbox,
            "preceding_text": table.context.preceding_text,
            "following_text": table.context.following_text,
        }
        
        return ProcessedTable(
            id=table.id,
            content=summary,
            page=table.page,
            document_context=self.document_context,
            metadata=metadata,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def process_json_file(
    json_path: Path,
    generate_summaries: bool = True,
    config: Optional[TableConfig] = None,
) -> List[ProcessedTable]:
    """Process a Marker JSON file and extract tables.
    
    Args:
        json_path: Path to JSON file
        generate_summaries: Whether to generate LLM summaries
        config: Optional TableConfig
        
    Returns:
        List of ProcessedTable objects
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    source_name = json_path.stem
    processor = TableProcessor(config=config, source_name=source_name)
    
    return processor.process_json(data, generate_summaries)


def process_and_save(
    json_path: Path,
    output_path: Optional[Path] = None,
    generate_summaries: bool = True,
    config: Optional[TableConfig] = None,
) -> List[ProcessedTable]:
    """Process JSON file and save tables to JSON.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to output JSON file
        generate_summaries: Whether to generate LLM summaries
        config: Optional TableConfig
        
    Returns:
        List of ProcessedTable objects
    """
    tables = process_json_file(json_path, generate_summaries, config)
    
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}-table.json"
    
    # 构建新格式: {document: {...}, tables: [...]}
    doc_context = tables[0].document_context if tables else DocumentContext()
    
    output_data = {
        "document": {
            "source": json_path.stem,
            "company": doc_context.company_short,
            "company_full": doc_context.company_name,
            "stock_code": doc_context.stock_code,
            "report_period": doc_context.report_period,
            "report_type": doc_context.report_type,
            "fiscal_year": doc_context.fiscal_year
        },
        "tables": [table.to_dict() for table in tables]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(tables)} tables to {output_path}")
    
    return tables


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for table processing."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Extract and summarize tables from Marker JSON output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with LLM summarization (using Ollama qwen3 by default)
  python process_table.py input.json -o tables.json
  
  # Extract only (no LLM)
  python process_table.py input.json -o tables.json --extract-only
  
Environment variables:
  TABLE_LLM_BASE_URL  - LLM API base URL (default: uses LLM_API_BASE or https://dashscope.aliyuncs.com/compatible-mode/v1)
  TABLE_LLM_API_KEY   - API key (default: uses LLM_API_KEY or DASHSCOPE_API_KEY)
  TABLE_LLM_MODEL     - Model name (default: uses LLM_MODEL or deepseek-v3.2)
        """
    )
    
    parser.add_argument("input", type=str, help="Input Marker JSON file path")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path")
    parser.add_argument("--extract-only", action="store_true", help="Skip LLM summarization")
    parser.add_argument("--base-url", type=str, help="LLM API base URL")
    parser.add_argument("--api-key", type=str, help="LLM API key")
    parser.add_argument("--model", type=str, help="LLM model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else None
    
    config = None
    if args.base_url or args.api_key or args.model:
        config = TableConfig.create(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
    
    if args.verbose:
        print(f"Input: {input_path}")
        print(f"Generate summaries: {not args.extract_only}")
    
    tables = process_and_save(
        input_path,
        output_path,
        generate_summaries=not args.extract_only,
        config=config,
    )
    
    # Print summary
    print(f"\n=== Table Summary ===")
    print(f"Total tables: {len(tables)}")
    for table in tables[:5]:
        section_path = table.metadata.get('section_path', [])
        section = " > ".join(section_path) if section_path else 'N/A'
        page = table.page + 1  # 转为1-based显示
        print(f"\n[{table.id}] Page {page}")
        print(f"  Section: {section}")
        if table.content:
            preview = table.content[:100].replace('\n', ' ')
            print(f"  Summary: {preview}...")
    
    if len(tables) > 5:
        print(f"\n... and {len(tables) - 5} more tables")


if __name__ == "__main__":
    main()
