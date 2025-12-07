"""Utility functions for rag_agent."""
from .logging import get_logger, TraceAdapter, ColorFormatter, set_logging_debug_mode, is_logging_debug_mode
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

__all__ = [
    # Logging
    "get_logger",
    "TraceAdapter",
    "ColorFormatter",
    "set_logging_debug_mode",
    "is_logging_debug_mode",

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
]