"""Complete preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .pdf_processor import PDFProcessor, PDFDocument
from .table_enhancer import TableEnhancer, EnhancedTable
from .chunker import SemanticChunker, TextChunk


@dataclass
class ProcessedChunk:
    """A chunk ready for embedding and indexing."""
    page_content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }


@dataclass
class PipelineResult:
    """Result of the preprocessing pipeline."""
    source_path: str
    source_name: str
    text_chunks: List[ProcessedChunk]
    table_chunks: List[ProcessedChunk]
    all_chunks: List[ProcessedChunk]
    stats: Dict[str, Any] = field(default_factory=dict)


class PreprocessPipeline:
    """Complete pipeline for preprocessing PDFs into chunks.
    
    This pipeline:
    1. Extracts text and tables from PDFs
    2. Enhances tables with semantic information
    3. Chunks text with semantic boundaries
    4. Generates consistent metadata and content hashes
    5. Outputs JSONL files ready for embedding
    
    Usage:
        pipeline = PreprocessPipeline()
        result = pipeline.process_pdf("path/to/file.pdf")
        pipeline.save_results(result, "outputs/")
    """
    
    def __init__(
        self,
        # PDF processing options
        context_lines_before: int = 5,
        context_lines_after: int = 3,
        min_table_rows: int = 2,
        min_table_cols: int = 2,
        # Text chunking options
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        # Table enhancement options
        include_raw_table_data: bool = False,
    ):
        """Initialize the preprocessing pipeline.
        
        Args:
            context_lines_before: Lines of context to capture before tables
            context_lines_after: Lines of context to capture after tables
            min_table_rows: Minimum rows for valid table
            min_table_cols: Minimum columns for valid table
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between text chunks
            min_chunk_size: Minimum size for text chunks
            include_raw_table_data: Include raw table data in output
        """
        self.pdf_processor = PDFProcessor(
            context_lines_before=context_lines_before,
            context_lines_after=context_lines_after,
            min_table_rows=min_table_rows,
            min_table_cols=min_table_cols,
        )
        
        self.table_enhancer = TableEnhancer(
            include_raw_data=include_raw_table_data,
        )
        
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
    
    def process_pdf(self, pdf_path: str) -> PipelineResult:
        """Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PipelineResult with all processed chunks
        """
        source_name = os.path.basename(pdf_path)
        
        # Step 1: Extract content from PDF
        doc = self.pdf_processor.process(pdf_path)
        
        # Step 2: Enhance tables
        enhanced_tables: List[EnhancedTable] = []
        for table in doc.tables:
            enhanced = self.table_enhancer.enhance(table, source_name)
            enhanced_tables.append(enhanced)
        
        # Step 3: Chunk text
        text_chunks = self.chunker.chunk_extracted_texts(
            doc.texts,
            source_name=source_name,
        )
        
        # Step 4: Convert to ProcessedChunks
        processed_text_chunks = self._convert_text_chunks(text_chunks, pdf_path)
        processed_table_chunks = self._convert_table_chunks(enhanced_tables, pdf_path)
        
        # Combine all chunks
        all_chunks = processed_text_chunks + processed_table_chunks
        
        # Compute stats
        stats = {
            "total_pages": doc.total_pages,
            "text_chunks": len(processed_text_chunks),
            "table_chunks": len(processed_table_chunks),
            "total_chunks": len(all_chunks),
            "tables_extracted": len(doc.tables),
            "tables_enhanced": len(enhanced_tables),
        }
        
        return PipelineResult(
            source_path=pdf_path,
            source_name=source_name,
            text_chunks=processed_text_chunks,
            table_chunks=processed_table_chunks,
            all_chunks=all_chunks,
            stats=stats,
        )
    
    def process_directory(
        self,
        input_dir: str,
        file_pattern: str = "*.pdf",
    ) -> List[PipelineResult]:
        """Process all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDFs
            file_pattern: Glob pattern for files
            
        Returns:
            List of PipelineResult objects
        """
        import glob
        
        pdf_files = glob.glob(os.path.join(input_dir, file_pattern))
        results = []
        
        for pdf_path in sorted(pdf_files):
            print(f"Processing: {pdf_path}")
            try:
                result = self.process_pdf(pdf_path)
                results.append(result)
                print(f"  → {result.stats['total_chunks']} chunks "
                      f"({result.stats['text_chunks']} text, "
                      f"{result.stats['table_chunks']} tables)")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        return results
    
    def _convert_text_chunks(
        self,
        chunks: List[TextChunk],
        source_path: str,
    ) -> List[ProcessedChunk]:
        """Convert TextChunks to ProcessedChunks."""
        processed = []
        
        for chunk in chunks:
            # Compute content hash
            content_sha1 = hashlib.sha1(chunk.content.encode("utf-8")).hexdigest()
            
            metadata = {
                **chunk.metadata,
                "source_path": source_path,
                "content_sha1": content_sha1,
            }
            
            processed.append(ProcessedChunk(
                page_content=chunk.content,
                metadata=metadata,
            ))
        
        return processed
    
    def _convert_table_chunks(
        self,
        enhanced_tables: List[EnhancedTable],
        source_path: str,
    ) -> List[ProcessedChunk]:
        """Convert EnhancedTables to ProcessedChunks."""
        processed = []
        
        for table in enhanced_tables:
            # Compute content hash
            content_sha1 = hashlib.sha1(
                table.enhanced_content.encode("utf-8")
            ).hexdigest()
            
            metadata = {
                **table.metadata,
                "source_path": source_path,
                "content_sha1": content_sha1,
                "semantic_description": table.semantic_description,
            }
            
            processed.append(ProcessedChunk(
                page_content=table.enhanced_content,
                metadata=metadata,
            ))
        
        return processed
    
    @staticmethod
    def save_results(
        results: List[PipelineResult] | PipelineResult,
        output_dir: str,
        save_separate: bool = True,
    ) -> Dict[str, str]:
        """Save processing results to JSONL files.
        
        Args:
            results: Single result or list of results
            output_dir: Output directory
            save_separate: Save text and table chunks separately
            
        Returns:
            Dictionary of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(results, PipelineResult):
            results = [results]
        
        # Collect all chunks
        all_text_chunks = []
        all_table_chunks = []
        all_chunks = []
        
        for result in results:
            all_text_chunks.extend(result.text_chunks)
            all_table_chunks.extend(result.table_chunks)
            all_chunks.extend(result.all_chunks)
        
        output_paths = {}
        
        # Save combined file
        all_chunks_path = os.path.join(output_dir, "all_chunks.jsonl")
        PreprocessPipeline._save_jsonl(all_chunks, all_chunks_path)
        output_paths["all_chunks"] = all_chunks_path
        
        if save_separate:
            # Save text chunks
            text_path = os.path.join(output_dir, "text_chunks.jsonl")
            PreprocessPipeline._save_jsonl(all_text_chunks, text_path)
            output_paths["text_chunks"] = text_path
            
            # Save table chunks
            table_path = os.path.join(output_dir, "table_chunks.jsonl")
            PreprocessPipeline._save_jsonl(all_table_chunks, table_path)
            output_paths["table_chunks"] = table_path
        
        # Print summary
        print(f"\n📁 Saved to {output_dir}/")
        print(f"   - all_chunks.jsonl: {len(all_chunks)} chunks")
        if save_separate:
            print(f"   - text_chunks.jsonl: {len(all_text_chunks)} chunks")
            print(f"   - table_chunks.jsonl: {len(all_table_chunks)} chunks")
        
        return output_paths
    
    @staticmethod
    def _save_jsonl(chunks: List[ProcessedChunk], path: str) -> None:
        """Save chunks to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                f.write(line + "\n")


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess PDFs for RAG knowledge base"
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default="data",
        help="Input directory containing PDFs",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="outputs",
        help="Output directory for processed chunks",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Target chunk size in characters",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--context_lines",
        type=int,
        default=5,
        help="Lines of context around tables",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📚 RAG Knowledge Base Preprocessor")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    print()
    
    # Initialize pipeline
    pipeline = PreprocessPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        context_lines_before=args.context_lines,
        context_lines_after=args.context_lines // 2,
    )
    
    # Process all PDFs
    results = pipeline.process_directory(args.input_dir)
    
    if not results:
        print("⚠️  No PDFs processed!")
        return
    
    # Save results
    pipeline.save_results(results, args.output_dir)
    
    # Print summary
    total_stats = {
        "files": len(results),
        "text_chunks": sum(r.stats["text_chunks"] for r in results),
        "table_chunks": sum(r.stats["table_chunks"] for r in results),
        "total_chunks": sum(r.stats["total_chunks"] for r in results),
    }
    
    print()
    print("=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Files processed: {total_stats['files']}")
    print(f"Text chunks:     {total_stats['text_chunks']}")
    print(f"Table chunks:    {total_stats['table_chunks']}")
    print(f"Total chunks:    {total_stats['total_chunks']}")
    print()
    print("✅ Done!")


if __name__ == "__main__":
    main()
