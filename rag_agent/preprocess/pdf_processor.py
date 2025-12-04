"""PDF processing with enhanced table extraction."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Suppress pdfplumber/pdfminer warnings about invalid color values
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*invalid float value.*")
warnings.filterwarnings("ignore", message=".*Cannot set.*stroke color.*")

# PDF parsing libraries
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None  # type: ignore
    _PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    _PYPDF2_AVAILABLE = True
except ImportError:
    PdfReader = None  # type: ignore
    _PYPDF2_AVAILABLE = False


@dataclass
class ExtractedTable:
    """Represents an extracted table with metadata."""
    content: str  # Markdown format
    raw_data: List[List[str]]  # Raw cell data
    columns: List[str]  # Column headers
    row_count: int
    page: int
    table_index: int  # Index on page
    bbox: Optional[Tuple[float, float, float, float]] = None
    title: str = ""
    context_before: str = ""  # Text before table
    context_after: str = ""  # Text after table
    unit: str = ""
    time_point: str = ""


@dataclass
class ExtractedText:
    """Represents extracted text chunk."""
    content: str
    page: int
    section: str = ""
    is_header: bool = False


@dataclass 
class PDFDocument:
    """Processed PDF document."""
    source_path: str
    source_name: str
    total_pages: int
    texts: List[ExtractedText] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFProcessor:
    """Enhanced PDF processor with table structure preservation.
    
    Features:
    - Real column header extraction
    - Context capture around tables
    - Unit and time point detection
    - Section/chapter tracking
    """
    
    # Patterns for detecting table metadata
    UNIT_PATTERNS = [
        (r"人民币?百万元", "百万元"),
        (r"人民币?亿元", "亿元"),
        (r"人民币?万元", "万元"),
        (r"人民币?千?元", "元"),
        (r"百分比|%|％", "%"),
        (r"百分点|个百分点|bp|bps", "百分点"),
    ]
    
    TIME_PATTERNS = [
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        r"(\d{4})\s*年\s*(第?[一二三四1-4]季度|[1-4]Q|Q[1-4])",
        r"(\d{4})\s*年\s*(\d{1,2})\s*月",
        r"(\d{4})\s*年?(上半年|下半年|全年|年度)",
        r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})",
        r"(\d{4})[/\-](\d{1,2})",
    ]
    
    # Section header patterns (Chinese financial reports)
    SECTION_PATTERNS = [
        r"^[一二三四五六七八九十]+[、.．]\s*(.+)$",
        r"^第[一二三四五六七八九十]+[章节部分]\s*(.+)$",
        r"^[\d]+[、.．]\s*(.+)$",
        r"^[（\(][一二三四五六七八九十\d]+[）\)]\s*(.+)$",
    ]
    
    def __init__(
        self,
        context_lines_before: int = 5,
        context_lines_after: int = 3,
        min_table_rows: int = 2,
        min_table_cols: int = 2,
    ):
        self.context_lines_before = context_lines_before
        self.context_lines_after = context_lines_after
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        
        if not _PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is required. Install: pip install pdfplumber")
    
    def process(self, pdf_path: str) -> PDFDocument:
        """Process a PDF file and extract text and tables.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFDocument with extracted content
        """
        source_name = os.path.basename(pdf_path)
        
        doc = PDFDocument(
            source_path=pdf_path,
            source_name=source_name,
            total_pages=0,
        )
        
        current_section = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            doc.total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                page_idx = page_num + 1  # 1-indexed
                
                # Extract full page text
                page_text = page.extract_text() or ""
                page_lines = page_text.split("\n")
                
                # Detect section headers
                for line in page_lines:
                    section = self._detect_section(line.strip())
                    if section:
                        current_section = section
                
                # Extract tables with context
                tables = page.extract_tables() or []
                table_bboxes = []
                
                for tbl_idx, table_data in enumerate(tables):
                    if not self._is_valid_table(table_data):
                        continue
                    
                    # Get table bbox if available
                    bbox = None
                    try:
                        page_tables = page.find_tables()
                        if tbl_idx < len(page_tables):
                            bbox = page_tables[tbl_idx].bbox
                            table_bboxes.append(bbox)
                    except Exception:
                        pass
                    
                    # Extract table with enhanced metadata
                    extracted = self._process_table(
                        table_data=table_data,
                        page_text=page_text,
                        page_lines=page_lines,
                        page_num=page_idx,
                        table_index=tbl_idx,
                        bbox=bbox,
                        section=current_section,
                    )
                    
                    if extracted:
                        doc.tables.append(extracted)
                
                # Extract text chunks (excluding table regions)
                text_chunks = self._extract_text_chunks(
                    page_text=page_text,
                    page_num=page_idx,
                    section=current_section,
                    table_bboxes=table_bboxes,
                )
                doc.texts.extend(text_chunks)
        
        return doc
    
    def _is_valid_table(self, table_data: List[List]) -> bool:
        """Check if table meets minimum requirements."""
        if not table_data:
            return False
        row_count = len(table_data)
        col_count = max(len(row) for row in table_data) if table_data else 0
        return row_count >= self.min_table_rows and col_count >= self.min_table_cols
    
    def _process_table(
        self,
        table_data: List[List],
        page_text: str,
        page_lines: List[str],
        page_num: int,
        table_index: int,
        bbox: Optional[Tuple],
        section: str,
    ) -> Optional[ExtractedTable]:
        """Process a single table with enhanced metadata extraction."""
        
        # Clean table data
        cleaned_data = self._clean_table_data(table_data)
        if not cleaned_data:
            return None
        
        # Extract column headers (intelligent detection)
        columns, data_rows = self._extract_columns(cleaned_data)
        
        # Detect title from context
        title = self._extract_table_title(page_lines, table_data, bbox)
        
        # Extract context before/after table
        context_before, context_after = self._extract_context(
            page_lines, table_data, bbox
        )
        
        # Detect unit and time point
        combined_context = f"{title} {context_before} {' '.join(columns)}"
        unit = self._detect_unit(combined_context)
        time_point = self._detect_time_point(combined_context)
        
        # Convert to markdown with real headers
        markdown = self._table_to_markdown(columns, data_rows)
        
        return ExtractedTable(
            content=markdown,
            raw_data=cleaned_data,
            columns=columns,
            row_count=len(data_rows),
            page=page_num,
            table_index=table_index,
            bbox=bbox,
            title=title,
            context_before=context_before,
            context_after=context_after,
            unit=unit,
            time_point=time_point,
        )
    
    def _clean_table_data(self, table_data: List[List]) -> List[List[str]]:
        """Clean and normalize table cell data."""
        cleaned = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean cell: remove extra whitespace, normalize
                    text = str(cell).strip()
                    text = re.sub(r'\s+', ' ', text)
                    cleaned_row.append(text)
            cleaned.append(cleaned_row)
        return cleaned
    
    def _extract_columns(
        self, 
        cleaned_data: List[List[str]]
    ) -> Tuple[List[str], List[List[str]]]:
        """Extract column headers from table data.
        
        Uses heuristics to detect header rows:
        1. First row if it contains non-numeric values
        2. Multiple header rows if detected
        """
        if not cleaned_data:
            return [], []
        
        # Check if first row looks like headers
        first_row = cleaned_data[0]
        
        def is_header_cell(cell: str) -> bool:
            """Check if cell looks like a header (not purely numeric)."""
            if not cell:
                return False
            # Remove common numeric patterns
            cleaned = re.sub(r'[\d,.\-%+()（）/／\s]+', '', cell)
            return len(cleaned) > 0
        
        header_count = sum(1 for cell in first_row if is_header_cell(cell))
        total_cells = len([c for c in first_row if c])
        
        # If >50% cells look like headers, use as column names
        if total_cells > 0 and header_count / total_cells > 0.5:
            columns = first_row
            data_rows = cleaned_data[1:]
            
            # Check for multi-row headers (second row also header-like)
            if data_rows:
                second_row = data_rows[0]
                second_header_count = sum(1 for cell in second_row if is_header_cell(cell))
                second_total = len([c for c in second_row if c])
                
                if second_total > 0 and second_header_count / second_total > 0.5:
                    # Merge header rows
                    merged_columns = []
                    for i in range(max(len(columns), len(second_row))):
                        c1 = columns[i] if i < len(columns) else ""
                        c2 = second_row[i] if i < len(second_row) else ""
                        if c1 and c2:
                            merged_columns.append(f"{c1} {c2}".strip())
                        else:
                            merged_columns.append(c1 or c2)
                    columns = merged_columns
                    data_rows = data_rows[1:]
        else:
            # Generate default column names
            col_count = max(len(row) for row in cleaned_data)
            columns = [f"列{i+1}" for i in range(col_count)]
            data_rows = cleaned_data
        
        # Ensure column names are unique
        seen = {}
        unique_columns = []
        for col in columns:
            if not col:
                col = "未命名"
            if col in seen:
                seen[col] += 1
                unique_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_columns.append(col)
        
        return unique_columns, data_rows
    
    def _extract_table_title(
        self,
        page_lines: List[str],
        table_data: List[List],
        bbox: Optional[Tuple],
    ) -> str:
        """Extract table title from surrounding text."""
        # Look for title patterns in lines before table
        title_candidates = []
        
        # Find approximate table start position
        if table_data and table_data[0]:
            first_cell = str(table_data[0][0] or "")
            for i, line in enumerate(page_lines):
                if first_cell and first_cell[:10] in line:
                    # Look at lines before this
                    start_idx = max(0, i - self.context_lines_before)
                    for j in range(start_idx, i):
                        candidate = page_lines[j].strip()
                        if self._looks_like_title(candidate):
                            title_candidates.append(candidate)
                    break
        
        # Return best candidate (usually last one before table)
        if title_candidates:
            return title_candidates[-1]
        return ""
    
    def _looks_like_title(self, text: str) -> bool:
        """Check if text looks like a table title."""
        if not text or len(text) < 3 or len(text) > 100:
            return False
        
        # Title patterns
        title_patterns = [
            r"表\s*\d",
            r"表[一二三四五六七八九十]",
            r"单位[:：]",
            r"人民币",
            r"百万元|亿元|万元",
            r"百分比|比例|指标",
            r"资产|负债|收入|支出|利润",
            r"合并|母公司|本行|本公司|本集团",
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _extract_context(
        self,
        page_lines: List[str],
        table_data: List[List],
        bbox: Optional[Tuple],
    ) -> Tuple[str, str]:
        """Extract text context before and after table."""
        context_before = ""
        context_after = ""
        
        if not table_data or not table_data[0]:
            return context_before, context_after
        
        first_cell = str(table_data[0][0] or "")[:20]
        last_row = table_data[-1]
        last_cell = str(last_row[-1] if last_row else "")[:20]
        
        table_start_idx = -1
        table_end_idx = -1
        
        for i, line in enumerate(page_lines):
            if table_start_idx < 0 and first_cell and first_cell in line:
                table_start_idx = i
            if last_cell and last_cell in line:
                table_end_idx = i
        
        if table_start_idx >= 0:
            start = max(0, table_start_idx - self.context_lines_before)
            context_before = "\n".join(page_lines[start:table_start_idx])
        
        if table_end_idx >= 0:
            end = min(len(page_lines), table_end_idx + self.context_lines_after + 1)
            context_after = "\n".join(page_lines[table_end_idx + 1:end])
        
        return context_before.strip(), context_after.strip()
    
    def _detect_unit(self, text: str) -> str:
        """Detect unit from text."""
        for pattern, unit in self.UNIT_PATTERNS:
            if re.search(pattern, text):
                return unit
        return ""
    
    def _detect_time_point(self, text: str) -> str:
        """Detect time point from text."""
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""
    
    def _detect_section(self, line: str) -> str:
        """Detect if line is a section header."""
        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, line)
            if match:
                return line
        return ""
    
    def _table_to_markdown(
        self, 
        columns: List[str], 
        data_rows: List[List[str]]
    ) -> str:
        """Convert table to markdown format."""
        lines = []
        
        # Header row
        header = " | ".join(columns)
        lines.append(header)
        
        # Separator
        separator = " | ".join(["---"] * len(columns))
        lines.append(separator)
        
        # Data rows
        for row in data_rows:
            # Pad row if needed
            padded = row + [""] * (len(columns) - len(row))
            row_str = " | ".join(padded[:len(columns)])
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def _extract_text_chunks(
        self,
        page_text: str,
        page_num: int,
        section: str,
        table_bboxes: List[Tuple],
    ) -> List[ExtractedText]:
        """Extract non-table text chunks from page."""
        # For simplicity, return full page text as one chunk
        # More sophisticated implementations can exclude table regions
        
        if not page_text.strip():
            return []
        
        # Check if line looks like a header
        lines = page_text.split("\n")
        is_header = any(self._detect_section(line.strip()) for line in lines[:3])
        
        return [ExtractedText(
            content=page_text.strip(),
            page=page_num,
            section=section,
            is_header=is_header,
        )]
