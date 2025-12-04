"""Semantic chunking for text content."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .pdf_processor import ExtractedText


@dataclass
class TextChunk:
    """A semantically coherent text chunk."""
    content: str
    page: int
    section: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# Mapping from PDF filename patterns to entity names for context injection
# This helps BM25 retrieval by associating each chunk with its source entity
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
    """Extract entity name from source filename.
    
    Args:
        source_name: PDF filename like 'CITIC-2025-h1.pdf'
        
    Returns:
        Entity name like '中信银行' or empty string if not found
    """
    if not source_name:
        return ""
    
    # Check each pattern
    upper_name = source_name.upper()
    for pattern, entity in SOURCE_ENTITY_MAP.items():
        if pattern.upper() in upper_name:
            return entity
    
    return ""


class SemanticChunker:
    """Semantic-aware text chunker.
    
    Features:
    - Respects sentence and paragraph boundaries
    - Preserves section structure
    - Overlapping chunks for context continuity
    - Special handling for financial report sections
    - Injects source entity name for better retrieval
    """
    
    # Section patterns that should start new chunks
    SECTION_BREAK_PATTERNS = [
        r"^[一二三四五六七八九十]+[、.．]\s*",
        r"^第[一二三四五六七八九十]+[章节部分]",
        r"^[\d]+[、.．]\s*",
        r"^[（\(][一二三四五六七八九十\d]+[）\)]",
    ]
    
    # Sentence ending patterns
    SENTENCE_END = r"[。！？；\n]"
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        respect_sentences: bool = True,
        preserve_sections: bool = True,
    ):
        """Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (won't split smaller)
            respect_sentences: Try to end chunks at sentence boundaries
            preserve_sections: Don't split across major sections
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        self.preserve_sections = preserve_sections
    
    def chunk_text(
        self,
        text: str,
        page: int = 0,
        section: str = "",
        source_name: str = "",
    ) -> List[TextChunk]:
        """Chunk a piece of text.
        
        Args:
            text: The text to chunk
            page: Page number
            section: Section name
            source_name: Source document name
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [TextChunk(
                content=text,
                page=page,
                section=section,
                chunk_index=0,
                metadata=self._build_metadata(text, source_name, page, section, 0),
            )]
        
        # Split into sections first if preserving sections
        if self.preserve_sections:
            sections = self._split_sections(text)
        else:
            sections = [(section, text)]
        
        chunks = []
        global_index = 0
        
        for sec_name, sec_text in sections:
            sec_chunks = self._chunk_section(
                sec_text, 
                sec_name or section,
                page,
                source_name,
                global_index,
            )
            chunks.extend(sec_chunks)
            global_index += len(sec_chunks)
        
        return chunks
    
    def _split_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text into sections based on headers."""
        lines = text.split("\n")
        sections = []
        current_section = ""
        current_lines = []
        
        for line in lines:
            is_header = any(
                re.match(pattern, line.strip()) 
                for pattern in self.SECTION_BREAK_PATTERNS
            )
            
            if is_header:
                # Save previous section
                if current_lines:
                    sections.append((current_section, "\n".join(current_lines)))
                current_section = line.strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        
        # Save last section
        if current_lines:
            sections.append((current_section, "\n".join(current_lines)))
        
        return sections if sections else [("", text)]
    
    def _chunk_section(
        self,
        text: str,
        section: str,
        page: int,
        source_name: str,
        start_index: int,
    ) -> List[TextChunk]:
        """Chunk a single section of text."""
        chunks = []
        
        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            chunks.append(TextChunk(
                content=text.strip(),
                page=page,
                section=section,
                chunk_index=start_index,
                metadata=self._build_metadata(text, source_name, page, section, start_index),
            ))
            return chunks
        
        # Split into sentences/paragraphs first
        if self.respect_sentences:
            segments = self._split_sentences(text)
        else:
            segments = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        # Merge segments into chunks
        current_chunk = []
        current_length = 0
        chunk_index = start_index
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # If adding this segment exceeds chunk size
            if current_length + len(segment) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_text,
                    page=page,
                    section=section,
                    chunk_index=chunk_index,
                    metadata=self._build_metadata(chunk_text, source_name, page, section, chunk_index),
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_segments = self._get_overlap_segments(current_chunk)
                current_chunk = overlap_segments + [segment]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(segment)
                current_length += len(segment)
        
        # Save final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    content=chunk_text,
                    page=page,
                    section=section,
                    chunk_index=chunk_index,
                    metadata=self._build_metadata(chunk_text, source_name, page, section, chunk_index),
                ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence endings
        sentences = re.split(self.SENTENCE_END, text)
        
        # Restore sentence endings
        result = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if sent:
                # Add back punctuation if not the last segment
                if i < len(sentences) - 1:
                    # Find original ending
                    match = re.search(self.SENTENCE_END, text[text.find(sent) + len(sent):])
                    if match:
                        sent += match.group(0)
                result.append(sent)
        
        return result
    
    def _get_overlap_segments(self, segments: List[str]) -> List[str]:
        """Get segments for overlap from end of current chunk."""
        if not segments or self.chunk_overlap <= 0:
            return []
        
        overlap = []
        total_length = 0
        
        for seg in reversed(segments):
            if total_length >= self.chunk_overlap:
                break
            overlap.insert(0, seg)
            total_length += len(seg)
        
        return overlap
    
    def _build_metadata(
        self,
        text: str,
        source_name: str,
        page: int,
        section: str,
        chunk_index: int,
    ) -> Dict[str, Any]:
        """Build metadata for a chunk."""
        # Compute content hash
        content_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
        
        return {
            "source": source_name,
            "page": page,
            "section": section,
            "chunk_index": chunk_index,
            "doctype": "text",
            "char_count": len(text),
            "content_sha1": content_sha1,
        }
    
    def chunk_extracted_texts(
        self,
        texts: List[ExtractedText],
        source_name: str = "",
        inject_source_entity: bool = True,
    ) -> List[TextChunk]:
        """Chunk multiple ExtractedText objects.
        
        Args:
            texts: List of ExtractedText objects to chunk
            source_name: Source document name
            inject_source_entity: Whether to inject entity name into chunk content
                                  for better BM25 retrieval
        """
        all_chunks = []
        global_index = 0
        
        # Extract entity name from source for injection
        entity_name = extract_entity_from_source(source_name) if inject_source_entity else ""
        
        for ext_text in texts:
            chunks = self.chunk_text(
                text=ext_text.content,
                page=ext_text.page,
                section=ext_text.section,
                source_name=source_name,
            )
            
            # Update global indices and inject entity name
            for chunk in chunks:
                chunk.chunk_index = global_index
                chunk.metadata["chunk_index"] = global_index
                
                # Inject source entity name if not already present in content
                # This helps BM25 retrieval by associating data with its source
                if entity_name and entity_name not in chunk.content:
                    chunk.content = f"{entity_name} {chunk.content}"
                    chunk.metadata["injected_entity"] = entity_name
                
                global_index += 1
            
            all_chunks.extend(chunks)
        
        return all_chunks
