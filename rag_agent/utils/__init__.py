"""Utility functions for rag_agent."""

from .text import (
    tokenize_zh,
    compute_overlap_ratio,
    normalize_terms,
    normalize_numbers,
    SYNONYM_MAP,
)
from .logging import get_logger, TraceAdapter, ColorFormatter, set_logging_debug_mode, is_logging_debug_mode
from .debug import (
    DebugPrinter,
    get_debug_printer,
    set_debug_mode,
    is_debug_enabled,
)
from .tracing import (
    trace_step,
    trace_pipeline,
    trace_pipeline_stream,
    traceable_step,
    get_current_trace,
    is_tracing_enabled,
    PipelineTrace,
    StepTiming,
    TracedStreamIterator,
)
from .embed_and_store import (
    load_jsonl,
    to_documents,
    dedup_by_sha1,
    build_embeddings,
    build_in_memory_store,
    build_or_update_faiss,
    create_vector_store,
)

__all__ = [
    # Text processing
    "tokenize_zh",
    "compute_overlap_ratio",
    "normalize_terms",
    "normalize_numbers",
    "SYNONYM_MAP",
    # Logging
    "get_logger",
    "TraceAdapter",
    "ColorFormatter",
    "set_logging_debug_mode",
    "is_logging_debug_mode",
    # Debug
    "DebugPrinter",
    "get_debug_printer",
    "set_debug_mode",
    "is_debug_enabled",
    # Tracing
    "trace_step",
    "trace_pipeline",
    "trace_pipeline_stream",
    "traceable_step",
    "get_current_trace",
    "is_tracing_enabled",
    "PipelineTrace",
    "StepTiming",
    "TracedStreamIterator",
    # Embedding & storage
    "load_jsonl",
    "to_documents",
    "dedup_by_sha1",
    "build_embeddings",
    "build_in_memory_store",
    "build_or_update_faiss",
    "create_vector_store",
]