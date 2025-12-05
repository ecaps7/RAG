"""Text processing module for extracting and chunking text from Marker JSON output.

This module implements a two-stage splitting strategy:
1. Macro Splitting: Split by document structure (headers)
2. Micro Splitting: Recursively split long sections by length

Features:
- Preserves page information from JSON
- Respects section hierarchy
- Injects context (section path) into chunks for better retrieval
- Handles placeholders for tables and images
"""

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from html.parser import HTMLParser


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Block:
    """Represents a block from Marker JSON."""
    id: str
    block_type: str
    html: str
    page: int
    text: str = ""  # Extracted plain text
    section_hierarchy: Dict[str, str] = field(default_factory=dict)
    bbox: List[float] = field(default_factory=list)


@dataclass
class TextChunk:
    """A semantically coherent text chunk with metadata."""
    content: str  # Content with context prefix (for embedding)
    original_content: str  # Original content without prefix
    page: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# HTML Text Extractor
# ============================================================================

class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        # Add newline before block elements
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'br'):
            if self.text_parts and not self.text_parts[-1].endswith('\n'):
                self.text_parts.append('\n')
                
    def handle_endtag(self, tag):
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div'):
            if self.text_parts and not self.text_parts[-1].endswith('\n'):
                self.text_parts.append('\n')
        self.current_tag = None
        
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
    parser.feed(html)
    return parser.get_text()


def extract_header_text(html: str) -> str:
    """Extract header text from HTML (h1, h2, h3, etc.)."""
    # Match h1-h6 tags
    match = re.search(r'<h[1-6][^>]*>(.*?)</h[1-6]>', html, re.DOTALL | re.IGNORECASE)
    if match:
        inner = match.group(1)
        # Remove any inner tags like <br/>
        inner = re.sub(r'<[^>]+>', ' ', inner)
        return inner.strip()
    return extract_text_from_html(html)


# ============================================================================
# Source Entity Mapping (for BM25 retrieval enhancement)
# ============================================================================

SOURCE_ENTITY_MAP = {
    "CITIC": "中信银行",
    "CMB": "招商银行",
    "ICBC": "工商银行",
    "CCB": "建设银行",
    "ABC": "农业银行",
    "BOC": "中国银行",
    "PSBC": "邮储银行",
    "BOCOM": "交通银行",
    "CEB": "光大银行",
    "CMBC": "民生银行",
    "CIB": "兴业银行",
    "SPDB": "浦发银行",
    "HXB": "华夏银行",
    "PAB": "平安银行",
}


def extract_entity_from_source(source_name: str) -> str:
    """Extract entity name from source filename."""
    if not source_name:
        return ""
    upper_name = source_name.upper()
    for pattern, entity in SOURCE_ENTITY_MAP.items():
        if pattern.upper() in upper_name:
            return entity
    return ""


# ============================================================================
# JSON Parser - Extract Blocks from Marker JSON
# ============================================================================

class MarkerJSONParser:
    """Parse Marker JSON output and extract blocks with metadata."""
    
    # Block types to include as text content
    TEXT_BLOCK_TYPES = {'Text', 'SectionHeader', 'ListItem', 'Caption'}
    
    # Block types to replace with placeholders
    TABLE_BLOCK_TYPES = {'Table', 'TableOfContents'}
    IMAGE_BLOCK_TYPES = {'Picture', 'Figure'}
    
    # Block types to skip
    SKIP_BLOCK_TYPES = {'PageHeader', 'PageFooter'}
    
    def __init__(self):
        self.blocks: List[Block] = []
        self.tables: List[Block] = []
        self.images: List[Block] = []
        self.section_headers: Dict[str, str] = {}  # id -> header text
        self._table_counter = 0
        self._image_counter = 0
        
    def parse(self, json_data: dict) -> Tuple[List[Block], List[Block], List[Block]]:
        """Parse JSON and extract blocks.
        
        Returns:
            Tuple of (text_blocks, table_blocks, image_blocks)
        """
        self.blocks = []
        self.tables = []
        self.images = []
        self.section_headers = {}
        self._table_counter = 0
        self._image_counter = 0
        
        # Process each page
        for page_block in json_data.get('children', []):
            if page_block.get('block_type') == 'Page':
                self._process_page(page_block)
                
        return self.blocks, self.tables, self.images
    
    def _process_page(self, page_block: dict):
        """Process a single page and its children."""
        for child in page_block.get('children', []):
            self._process_block(child)
            
    def _process_block(self, block_data: dict):
        """Process a single block."""
        block_type = block_data.get('block_type', '')
        block_id = block_data.get('id', '')
        html = block_data.get('html', '')
        page = block_data.get('page', 0)
        section_hierarchy = block_data.get('section_hierarchy', {})
        bbox = block_data.get('bbox', [])
        
        # Skip certain block types
        if block_type in self.SKIP_BLOCK_TYPES:
            return
            
        # Extract text
        if block_type == 'SectionHeader':
            text = extract_header_text(html)
            # Store header text for section hierarchy resolution
            self.section_headers[block_id] = text
        else:
            text = extract_text_from_html(html)
            
        block = Block(
            id=block_id,
            block_type=block_type,
            html=html,
            page=page,
            text=text,
            section_hierarchy=section_hierarchy,
            bbox=bbox,
        )
        
        # Categorize block
        if block_type in self.TABLE_BLOCK_TYPES:
            block.text = f"{{{{TABLE_{self._table_counter}}}}}"
            self._table_counter += 1
            self.tables.append(block)
            self.blocks.append(block)  # Also add to blocks for position
        elif block_type in self.IMAGE_BLOCK_TYPES:
            block.text = f"{{{{IMAGE_{self._image_counter}}}}}"
            self._image_counter += 1
            self.images.append(block)
            self.blocks.append(block)  # Also add to blocks for position
        elif block_type in self.TEXT_BLOCK_TYPES:
            self.blocks.append(block)
            
    def resolve_section_path(self, block: Block) -> List[str]:
        """Resolve section hierarchy to a list of header texts."""
        path = []
        hierarchy = block.section_hierarchy
        
        # Sort by level (1, 2, 3, etc.)
        for level in sorted(hierarchy.keys(), key=lambda x: int(x)):
            header_id = hierarchy[level]
            if header_id in self.section_headers:
                path.append(self.section_headers[header_id])
                
        return path


# ============================================================================
# Text Chunker - Two-Stage Splitting Strategy
# ============================================================================

class TextChunker:
    """Semantic-aware text chunker with two-stage splitting.
    
    Stage 1: Macro Splitting - Group blocks by section
    Stage 2: Micro Splitting - Recursively split long sections
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        separators: List[str] = None,
    ):
        """Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size
            separators: Separators for recursive splitting (priority order)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        
    def chunk_blocks(
        self,
        blocks: List[Block],
        parser: MarkerJSONParser,
        source_name: str = "",
    ) -> List[TextChunk]:
        """Chunk blocks using two-stage splitting.
        
        Args:
            blocks: List of Block objects
            parser: MarkerJSONParser instance (for section resolution)
            source_name: Source document name
            
        Returns:
            List of TextChunk objects
        """
        if not blocks:
            return []
            
        # Stage 1: Group blocks by section
        sections = self._group_by_section(blocks, parser)
        
        # Stage 2: Chunk each section
        all_chunks = []
        entity_name = extract_entity_from_source(source_name)
        
        for section_path, section_blocks in sections:
            section_chunks = self._chunk_section(
                section_blocks, 
                section_path, 
                source_name,
                entity_name,
            )
            all_chunks.extend(section_chunks)
            
        return all_chunks
    
    def _group_by_section(
        self, 
        blocks: List[Block], 
        parser: MarkerJSONParser
    ) -> List[Tuple[List[str], List[Block]]]:
        """Group consecutive blocks by their section hierarchy."""
        if not blocks:
            return []
            
        sections = []
        current_path = None
        current_blocks = []
        
        for block in blocks:
            path = parser.resolve_section_path(block)
            
            # If section changed, save current section
            if path != current_path and current_blocks:
                sections.append((current_path or [], current_blocks))
                current_blocks = []
                
            current_path = path
            current_blocks.append(block)
            
        # Save last section
        if current_blocks:
            sections.append((current_path or [], current_blocks))
            
        return sections
    
    def _chunk_section(
        self,
        blocks: List[Block],
        section_path: List[str],
        source_name: str,
        entity_name: str,
    ) -> List[TextChunk]:
        """Chunk a single section using recursive character splitting."""
        if not blocks:
            return []
            
        # Combine block texts
        combined_text = "\n".join(b.text for b in blocks if b.text)
        if not combined_text.strip():
            return []
            
        # Get page range
        pages = [b.page for b in blocks]
        start_page = min(pages) if pages else 0
        end_page = max(pages) if pages else 0
        
        # Build context prefix
        context_prefix = ""
        if section_path:
            context_prefix = f"【章节: {' > '.join(section_path)}】\n"
        if entity_name:
            context_prefix = f"【来源: {entity_name}】" + context_prefix
            
        # If text is small enough, return as single chunk
        if len(combined_text) <= self.chunk_size:
            return [self._create_chunk(
                combined_text,
                context_prefix,
                start_page,
                end_page,
                section_path,
                source_name,
                entity_name,
            )]
            
        # Recursive splitting
        split_texts = self._recursive_split(combined_text)
        
        # Create chunks with overlap
        chunks = []
        for i, text in enumerate(split_texts):
            if len(text.strip()) < self.min_chunk_size:
                continue
            chunks.append(self._create_chunk(
                text,
                context_prefix,
                start_page,
                end_page,
                section_path,
                source_name,
                entity_name,
            ))
            
        return chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text using separators."""
        return self._split_with_separator(text, 0)
    
    def _split_with_separator(self, text: str, sep_index: int) -> List[str]:
        """Split text with separator at given index, fall back to next separator if needed."""
        if len(text) <= self.chunk_size:
            return [text]
            
        if sep_index >= len(self.separators):
            # No more separators, force split
            return self._force_split(text)
            
        separator = self.separators[sep_index]
        parts = text.split(separator)
        
        if len(parts) == 1:
            # Separator not found, try next
            return self._split_with_separator(text, sep_index + 1)
            
        # Merge parts into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for part in parts:
            part_with_sep = part + separator if part else separator
            part_len = len(part_with_sep)
            
            if current_length + part_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                
                # Recursively split if still too long
                if len(chunk_text) > self.chunk_size:
                    chunks.extend(self._split_with_separator(chunk_text, sep_index + 1))
                else:
                    chunks.append(chunk_text)
                    
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk, separator)
                current_chunk = [overlap_text, part] if overlap_text else [part]
                current_length = len(separator.join(current_chunk))
            else:
                current_chunk.append(part)
                current_length += part_len
                
        # Save last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.chunk_size:
                chunks.extend(self._split_with_separator(chunk_text, sep_index + 1))
            else:
                chunks.append(chunk_text)
                
        return chunks
    
    def _get_overlap(self, parts: List[str], separator: str) -> str:
        """Get overlap text from end of parts."""
        if not parts or self.chunk_overlap <= 0:
            return ""
            
        overlap_parts = []
        total_length = 0
        
        for part in reversed(parts):
            if total_length >= self.chunk_overlap:
                break
            overlap_parts.insert(0, part)
            total_length += len(part) + len(separator)
            
        return separator.join(overlap_parts)
    
    def _force_split(self, text: str) -> List[str]:
        """Force split text by character count."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        context_prefix: str,
        start_page: int,
        end_page: int,
        section_path: List[str],
        source_name: str,
        entity_name: str,
    ) -> TextChunk:
        """Create a TextChunk with metadata."""
        content = context_prefix + text if context_prefix else text
        content_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()
        
        return TextChunk(
            content=content,
            original_content=text,
            page=start_page,
            metadata={
                "type": "text",
                "source": source_name,
                "page": start_page,
                "page_end": end_page,
                "section_path": section_path,
                "section": " > ".join(section_path) if section_path else "",
                "entity": entity_name,
                "char_count": len(text),
                "content_sha1": content_hash,
            }
        )


# ============================================================================
# Main Processing Function
# ============================================================================

def process_json_file(
    json_path: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[TextChunk]:
    """Process a Marker JSON file and return text chunks.
    
    Args:
        json_path: Path to JSON file
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Parse blocks
    parser = MarkerJSONParser()
    blocks, tables, images = parser.parse(data)
    
    # Get source name from filename
    source_name = json_path.stem
    
    # Chunk text
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk_blocks(blocks, parser, source_name)
    
    return chunks


def process_and_save(
    json_path: Path,
    output_path: Optional[Path] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[TextChunk]:
    """Process JSON file and save chunks to JSON.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to output JSON file (default: same dir, -chunks.json suffix)
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    chunks = process_json_file(json_path, chunk_size, chunk_overlap)
    
    # Default output path
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}-text-chunks.json"
        
    # Convert to serializable format
    output_data = []
    for i, chunk in enumerate(chunks):
        output_data.append({
            "id": i,
            "content": chunk.content,
            "original_content": chunk.original_content,
            "page": chunk.page,
            **chunk.metadata,
        })
        
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(chunks)} chunks to {output_path}")
    
    return chunks


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Marker JSON and extract text chunks")
    parser.add_argument("input", type=str, help="Input JSON file path")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path")
    parser.add_argument("--chunk-size", type=int, default=800, help="Target chunk size (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    
    args = parser.parse_args()
    
    json_path = Path(args.input)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        exit(1)
        
    output_path = Path(args.output) if args.output else None
    
    chunks = process_and_save(
        json_path,
        output_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    # Print sample
    print(f"\n=== Sample Chunks ===")
    for chunk in chunks[:3]:
        print(f"\n--- Page {chunk.page} ---")
        print(f"Section: {chunk.metadata.get('section', 'N/A')}")
        print(f"Content preview: {chunk.content[:200]}...")
        print(f"Char count: {chunk.metadata.get('char_count')}")