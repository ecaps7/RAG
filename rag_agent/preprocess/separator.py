"""Content separator for splitting Markdown documents into text, tables, and images.

This module implements functionality to separate a Markdown document into:
1. Ordinary text (with tables and images replaced by placeholders)
2. Extracted tables
3. Extracted images
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class ExtractedItem:
    """An item extracted from the text (table or image)."""
    id: str
    type: str  # 'table' or 'image'
    content: str
    start: int
    end: int
    placeholder: str

@dataclass
class SeparatedContent:
    """Result of content separation."""
    text: str  # Text with placeholders
    tables: List[ExtractedItem]
    images: List[ExtractedItem]

class Separator:
    """Separates Markdown content into text, tables, and images."""

    # Regex patterns
    
    # Markdown table: starts with |, has separator row with dashes
    # Adapted from process_table.py
    MARKDOWN_TABLE_PATTERN = re.compile(
        r"(\|[^\n]+\|\n"           # Header row
        r"\|[-:\s|]+\|\n"          # Separator row (---|---)
        r"(?:\|[^\n]+\|\n?)+)",    # Data rows
        re.MULTILINE
    )
    
    # HTML table: <table>...</table>
    HTML_TABLE_PATTERN = re.compile(
        r"<table[^>]*>.*?</table>",
        re.DOTALL | re.IGNORECASE
    )

    # Markdown image: ![alt](src "title")
    # Captures: 1=alt, 2=src
    MARKDOWN_IMAGE_PATTERN = re.compile(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        re.MULTILINE
    )

    # HTML image: <img ...>
    HTML_IMAGE_PATTERN = re.compile(
        r"<img[^>]+>",
        re.IGNORECASE
    )

    def __init__(self):
        self._table_counter = 0
        self._image_counter = 0

    def process(self, content: str) -> SeparatedContent:
        """
        Process content and separate tables and images from text.
        
        Args:
            content: Markdown content
            
        Returns:
            SeparatedContent object containing text with placeholders and extracted items
        """
        # Reset counters for each process call if we want unique IDs per file processing
        # Or we can keep them if we want global uniqueness across calls (but usually per file is better)
        self._table_counter = 0
        self._image_counter = 0
        
        items: List[ExtractedItem] = []
        
        # 1. Find Tables
        # Markdown tables
        for match in self.MARKDOWN_TABLE_PATTERN.finditer(content):
            items.append(self._create_item(match, "table"))
            
        # HTML tables
        for match in self.HTML_TABLE_PATTERN.finditer(content):
            items.append(self._create_item(match, "table"))

        # 2. Find Images
        # Markdown images
        for match in self.MARKDOWN_IMAGE_PATTERN.finditer(content):
            items.append(self._create_item(match, "image"))
            
        # HTML images
        for match in self.HTML_IMAGE_PATTERN.finditer(content):
            items.append(self._create_item(match, "image"))

        # 3. Handle overlaps (e.g. images inside tables)
        # We want to keep the outer container (table) and ignore the inner item (image)
        # as a separate item, because it will be removed with the table.
        items = self._filter_overlaps(items)
        
        # 4. Replace with placeholders
        # We replace from end to start to avoid shifting indices
        result_text = content
        final_tables = []
        final_images = []
        
        # Sort by start position descending for replacement
        sorted_items = sorted(items, key=lambda x: x.start, reverse=True)
        
        for item in sorted_items:
            # Replace content with placeholder
            # Ensure we are replacing exactly the span
            result_text = result_text[:item.start] + item.placeholder + result_text[item.end:]
            
            if item.type == "table":
                final_tables.append(item)
            else:
                final_images.append(item)
                
        # Restore order (ascending by position) for output lists
        final_tables.sort(key=lambda x: x.start)
        final_images.sort(key=lambda x: x.start)
        
        return SeparatedContent(
            text=result_text,
            tables=final_tables,
            images=final_images
        )

    def _create_item(self, match: re.Match, type_: str) -> ExtractedItem:
        content = match.group(0)
        start, end = match.span()
        
        if type_ == "table":
            id_ = f"TABLE_{self._table_counter}"
            # Table regex consumes the final newline, so we add it back to preserve spacing
            placeholder = f"{{{{{id_}}}}}\n"
            self._table_counter += 1
        else:
            id_ = f"IMAGE_{self._image_counter}"
            placeholder = f"{{{{{id_}}}}}"
            self._image_counter += 1
            
        return ExtractedItem(
            id=id_,
            type=type_,
            content=content,
            start=start,
            end=end,
            placeholder=placeholder
        )

    def _filter_overlaps(self, items: List[ExtractedItem]) -> List[ExtractedItem]:
        """Filter out items that are contained within other items."""
        if not items:
            return []
            
        # Sort by start position
        items.sort(key=lambda x: x.start)
        
        kept_items = []
        
        for i, item in enumerate(items):
            is_contained = False
            for j, other in enumerate(items):
                if i == j:
                    continue
                # Check if item is strictly inside other
                # If item is inside other, we discard item (it's part of the other content)
                if item.start >= other.start and item.end <= other.end:
                    # If ranges are identical, we need a tie-breaker.
                    # Prioritize TABLE over IMAGE if they somehow match same text (unlikely)
                    # or just keep the first one encountered/processed.
                    # But here we are comparing all pairs.
                    
                    # If strictly inside or equal but different objects
                    if not (item.start == other.start and item.end == other.end):
                         is_contained = True
                         break
                    else:
                        # Exact match. 
                        # If i > j, we might have already processed j.
                        # Let's just say if we have exact duplicates, we keep one.
                        # But regexes for table and image shouldn't match exact same text usually.
                        pass
            
            if not is_contained:
                kept_items.append(item)
                
        return kept_items
    
def separate():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Separate Markdown content.")
    parser.add_argument("--input", "-i", required=True, help="Input Markdown file")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        exit(1)
        
    content = input_path.read_text(encoding="utf-8")
    separator = Separator()
    result = separator.process(content)
    
    print("=== Processed Text ===")
    print(result.text)
    return result

if __name__ == "__main__":
    separate()