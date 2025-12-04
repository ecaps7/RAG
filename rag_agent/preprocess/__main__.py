"""CLI entry point for preprocessing pipeline.

Usage:
    python -m rag_agent.preprocess --input data/ --output outputs/
    python -m rag_agent.preprocess --input data/招商银行2025年一季报.pdf --output outputs/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .pipeline import PreprocessPipeline


def main():
    """Main CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess PDF documents for RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in a directory
  python -m rag_agent.preprocess --input data/ --output outputs/
  
  # Process a single PDF file
  python -m rag_agent.preprocess --input data/report.pdf --output outputs/
  
  # Custom chunk settings
  python -m rag_agent.preprocess --input data/ --output outputs/ --chunk-size 800 --overlap 150
  
  # Verbose output
  python -m rag_agent.preprocess --input data/ --output outputs/ --verbose
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input path: directory containing PDFs or single PDF file"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for processed chunks and indexes"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Maximum chunk size in characters (default: 600)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters (default: 100)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--skip-vector-store",
        action="store_true",
        help="Skip building vector store (only generate chunks)"
    )
    
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip building BM25 index"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Validate input path
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Collect PDF files
    pdf_files: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            pdf_files = [input_path]
        else:
            print(f"❌ Error: Input file is not a PDF: {input_path}")
            sys.exit(1)
    else:
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ Error: No PDF files found in: {input_path}")
            sys.exit(1)
    
    print(f"📁 Found {len(pdf_files)} PDF file(s) to process")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    print("🔧 Initializing preprocessing pipeline...")
    pipeline = PreprocessPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    
    # Process all PDFs
    all_chunks = []
    text_chunks = []
    table_chunks = []
    
    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")
        print("-" * 50)
        
        try:
            result = pipeline.process_pdf(str(pdf_path))
            
            # PipelineResult contains text_chunks and table_chunks
            all_chunks.extend(result.all_chunks)
            text_chunks.extend(result.text_chunks)
            table_chunks.extend(result.table_chunks)
            
            if args.verbose:
                print(f"   ✓ Generated {result.stats['total_chunks']} chunks")
                print(f"     - Text chunks: {result.stats['text_chunks']}")
                print(f"     - Table chunks: {result.stats['table_chunks']}")
                
        except Exception as e:
            print(f"   ❌ Failed to process {pdf_path.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    if not all_chunks:
        print("\n❌ No chunks generated. Check input files.")
        sys.exit(1)
    
    # Save chunks
    print(f"\n💾 Saving {len(all_chunks)} chunks to {output_dir}")
    
    all_chunks_path = output_dir / "all_chunks.jsonl"
    text_chunks_path = output_dir / "text_chunks.jsonl"
    table_chunks_path = output_dir / "table_chunks.jsonl"
    
    with open(all_chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    print(f"   ✓ all_chunks.jsonl ({len(all_chunks)} chunks)")
    
    with open(text_chunks_path, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    print(f"   ✓ text_chunks.jsonl ({len(text_chunks)} chunks)")
    
    with open(table_chunks_path, "w", encoding="utf-8") as f:
        for chunk in table_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    print(f"   ✓ table_chunks.jsonl ({len(table_chunks)} chunks)")
    
    # Build BM25 index
    if not args.skip_bm25:
        print("\n🔍 Building BM25 index...")
        try:
            bm25_path = output_dir / "bm25_index.json"
            build_bm25_index(all_chunks, bm25_path)
            print(f"   ✓ bm25_index.json")
        except Exception as e:
            print(f"   ⚠️ Failed to build BM25 index: {e}")
    
    # Build vector store
    if not args.skip_vector_store:
        print("\n🧮 Building vector store...")
        try:
            vector_dir = output_dir / "vector_store"
            build_vector_store(all_chunks, vector_dir)
            print(f"   ✓ vector_store/index.faiss")
        except Exception as e:
            print(f"   ⚠️ Failed to build vector store: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Preprocessing complete!")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Text chunks:  {len(text_chunks)}")
    print(f"   Table chunks: {len(table_chunks)}")
    print(f"   Output dir:   {output_dir}")


def build_bm25_index(chunks: List, output_path: Path):
    """Build BM25 index from chunks (ProcessedChunk objects)."""
    import jieba
    
    # Build document list
    documents = []
    for chunk in chunks:
        # Handle ProcessedChunk objects
        if hasattr(chunk, 'page_content'):
            text = chunk.page_content
            metadata = chunk.metadata
        else:
            # Fallback for dict
            text = chunk.get("page_content", "") or chunk.get("text", "") or chunk.get("content", "")
            metadata = chunk.get("metadata", {})
        
        # Add semantic description if available
        if metadata.get("semantic_description"):
            text = metadata["semantic_description"] + " " + text
        documents.append(text)
    
    # Tokenize
    tokenized_docs = []
    for doc in documents:
        tokens = list(jieba.cut(doc))
        tokenized_docs.append(tokens)
    
    # Build BM25 data structure
    bm25_data = {
        "documents": [
            {
                "id": (chunk.metadata if hasattr(chunk, 'metadata') else chunk.get("metadata", {})).get("content_sha1", f"chunk_{i}"),
                "tokens": tokenized_docs[i],
                "source": (chunk.metadata if hasattr(chunk, 'metadata') else chunk.get("metadata", {})).get("source", ""),
                "title": (chunk.metadata if hasattr(chunk, 'metadata') else chunk.get("metadata", {})).get("table_name", ""),
            }
            for i, chunk in enumerate(chunks)
        ],
        "metadata": {
            "total_docs": len(chunks),
            "avg_doc_len": sum(len(t) for t in tokenized_docs) / len(tokenized_docs) if tokenized_docs else 0
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False, indent=2)


def build_vector_store(chunks: List, output_dir: Path):
    """Build FAISS vector store from chunks (ProcessedChunk objects)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare texts and metadata
    texts = []
    metadatas = []
    
    for chunk in chunks:
        # Handle ProcessedChunk objects
        if hasattr(chunk, 'page_content'):
            text = chunk.page_content
            chunk_meta = chunk.metadata
        else:
            # Fallback for dict
            text = chunk.get("page_content", "") or chunk.get("text", "") or chunk.get("content", "")
            chunk_meta = chunk.get("metadata", {})
        
        texts.append(text)
        
        # Build metadata for vector store
        meta = {
            "chunk_id": chunk_meta.get("content_sha1", ""),
            "source": chunk_meta.get("source", ""),
            "title": chunk_meta.get("table_name", ""),
            "doctype": chunk_meta.get("doctype", "text"),
            "page": chunk_meta.get("page", 0),
        }
        
        # Add table-specific metadata
        if chunk_meta.get("doctype") == "table":
            meta["columns"] = chunk_meta.get("columns", [])
            meta["row_count"] = chunk_meta.get("row_count", 0)
            meta["unit"] = chunk_meta.get("unit", "")
            meta["time_point"] = chunk_meta.get("time_point", "")
        
        metadatas.append(meta)
    
    # Use existing embed_and_store utility
    try:
        from ..utils.embed_and_store import build_faiss_store
        build_faiss_store(texts, metadatas, str(output_dir))
    except ImportError:
        # Fallback: use langchain directly with Ollama embeddings
        from langchain_community.vectorstores import FAISS
        
        # Try Ollama first, then fall back to HuggingFace
        try:
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(
                model="qwen3-embedding:0.6b",
                base_url="http://localhost:11434",
            )
        except ImportError:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                model_kwargs={"device": "cpu", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True}
            )
        
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        vectorstore.save_local(str(output_dir))


if __name__ == "__main__":
    main()
