"""Knowledge base preprocessing module.

This module provides enhanced PDF processing with:
- Structured table extraction with real column names
- Context-aware chunking
- Semantic metadata enrichment
- Table stream processing with "Summary + Source Code" dual-track approach
"""

from .process_table import (
    TableConfig,
    TableContext,
    ExtractedTable,
    ProcessedTable,
    TableSummarizer,
    TableProcessor,
)

__all__ = [
    "PDFProcessor",
    "TableEnhancer", 
    "SemanticChunker",
    "PreprocessPipeline",
    # Table processing
    "TableConfig",
    "TableFormat",
    "TableContext",
    "ExtractedTable",
    "ProcessedTable",
    "TableExtractor",
    "TableSummarizer",
    "TableProcessor",
    "create_table_processor",
    "extract_tables_from_markdown",
    "process_tables_with_summaries",
]
