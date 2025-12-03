"""Tracing utilities for observability with LangSmith integration."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Generator

# Try to import langsmith for tracing
try:
    from langsmith import traceable, trace as langsmith_trace
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_AVAILABLE = True
except ImportError:
    traceable = None
    langsmith_trace = None
    get_current_run_tree = None
    LANGSMITH_AVAILABLE = False


F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


@dataclass
class StepTiming:
    """Timing information for a pipeline step."""
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self):
        """Mark the step as finished and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass 
class PipelineTrace:
    """Complete trace of a RAG pipeline execution."""
    trace_id: str
    question: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    steps: List[StepTiming] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: StepTiming):
        """Add a completed step to the trace."""
        self.steps.append(step)
    
    def finish(self):
        """Mark the pipeline as finished."""
        self.end_time = time.perf_counter()
        self.total_duration_ms = (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for logging/export."""
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }


# Thread-local storage for current trace
_current_trace: Optional[PipelineTrace] = None


def get_current_trace() -> Optional[PipelineTrace]:
    """Get the current pipeline trace if any."""
    return _current_trace


def set_current_trace(trace: Optional[PipelineTrace]):
    """Set the current pipeline trace."""
    global _current_trace
    _current_trace = trace


def _is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled via environment variables."""
    # Support both LANGSMITH_TRACING and LANGCHAIN_TRACING_V2
    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return LANGSMITH_AVAILABLE and tracing_enabled and bool(api_key)


@contextmanager
def trace_step(name: str, **metadata) -> Generator[StepTiming, None, None]:
    """Context manager for tracing a pipeline step with timing.
    
    Usage:
        with trace_step("intent_classification", question=q) as step:
            result = classify(q)
            step.metadata["intent"] = result.intent
    """
    step = StepTiming(name=name, start_time=time.perf_counter(), metadata=metadata)
    try:
        yield step
    finally:
        step.finish()
        trace = get_current_trace()
        if trace:
            trace.add_step(step)


def trace_pipeline(run_name: str = "rag_pipeline"):
    """Decorator to trace an entire pipeline run.
    
    Integrates with LangSmith if available and configured.
    Uses langsmith.trace context manager to ensure all child calls
    (including LangChain auto-instrumentation) are nested under this parent.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract question from args (assuming first arg after self is question)
            question = ""
            if len(args) > 1:
                question = str(args[1]) if args[1] else ""
            elif "question" in kwargs:
                question = str(kwargs["question"])
            
            # Generate trace ID
            import uuid
            trace_id = kwargs.get("trace_id") or str(uuid.uuid4())[:8]
            
            # Create pipeline trace
            trace = PipelineTrace(trace_id=trace_id, question=question)
            set_current_trace(trace)
            
            try:
                # If LangSmith is available and configured, use trace context manager
                if _is_langsmith_enabled() and langsmith_trace is not None:
                    # Use langsmith.trace context manager for proper nesting
                    with langsmith_trace(
                        name=run_name,
                        run_type="chain",
                        inputs={"question": question},
                    ) as run:
                        result = func(*args, **kwargs)
                        # Update outputs if result has useful info
                        if hasattr(result, '__dict__'):
                            run.end(outputs={"result": str(result)[:500]})
                        return result
                else:
                    return func(*args, **kwargs)
            finally:
                trace.finish()
                set_current_trace(None)
        
        return wrapper  # type: ignore
    return decorator


def traceable_step(name: str, run_type: str = "chain"):
    """Decorator to make a function traceable with LangSmith and local timing.
    
    Args:
        name: Name of the step for tracing
        run_type: LangSmith run type (chain, llm, retriever, tool, etc.)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            step = StepTiming(name=name, start_time=time.perf_counter())
            
            try:
                # Execute with LangSmith tracing if available
                if _is_langsmith_enabled() and langsmith_trace is not None:
                    with langsmith_trace(
                        name=name,
                        run_type=run_type,
                    ):
                        result = func(*args, **kwargs)
                    return result
                else:
                    return func(*args, **kwargs)
            finally:
                step.finish()
                trace = get_current_trace()
                if trace:
                    trace.add_step(step)
        
        return wrapper  # type: ignore
    return decorator


def log_to_langsmith(metadata: Dict[str, Any], name: str = "custom_log"):
    """Log custom metadata to the current LangSmith run if available."""
    if not LANGSMITH_AVAILABLE:
        return
    
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata(metadata)
    except Exception:
        pass


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return _is_langsmith_enabled()


class TracedStreamIterator(Iterator[T]):
    """A wrapper iterator that keeps the LangSmith trace context alive while iterating.
    
    This is useful for streaming responses where the actual generation happens
    during iteration, not during the function call that creates the iterator.
    """
    
    def __init__(
        self,
        iterator: Iterator[T],
        run_context: Any,
        trace: Optional[PipelineTrace] = None,
        step_name: str = "stream_generation",
    ):
        self._iterator = iterator
        self._run_context = run_context
        self._trace = trace
        self._step_name = step_name
        self._step = StepTiming(name=step_name, start_time=time.perf_counter())
        self._exhausted = False
        self._collected_output: List[str] = []
    
    def __iter__(self) -> Iterator[T]:
        return self
    
    def __next__(self) -> T:
        try:
            value = next(self._iterator)
            # Collect output for tracing
            if isinstance(value, str):
                self._collected_output.append(value)
            return value
        except StopIteration:
            self._finish()
            raise
        except Exception as e:
            # Handle any other exceptions during iteration
            self._finish(error=e)
            raise
    
    def _finish(self, error: Optional[Exception] = None):
        """Clean up when iteration is complete."""
        if self._exhausted:
            return
        self._exhausted = True
        
        # Finish step timing
        self._step.finish()
        if self._trace:
            self._step.metadata["output_length"] = len("".join(self._collected_output))
            self._trace.add_step(self._step)
            self._trace.finish()
            set_current_trace(None)
        
        # End LangSmith run context properly using __exit__
        if self._run_context is not None:
            try:
                output_text = "".join(self._collected_output)
                # First set outputs on the run
                if hasattr(self._run_context, 'end'):
                    self._run_context.end(outputs={"result": output_text[:500] if output_text else ""})
                # Then properly exit the context manager
                if error is not None:
                    self._run_context.__exit__(type(error), error, error.__traceback__)
                else:
                    self._run_context.__exit__(None, None, None)
            except Exception:
                # Fallback: try to exit anyway
                try:
                    self._run_context.__exit__(None, None, None)
                except Exception:
                    pass
    
    def __del__(self):
        """Ensure cleanup happens even if iterator is not fully consumed."""
        self._finish()


def trace_pipeline_stream(run_name: str = "rag_pipeline_stream"):
    """Decorator to trace a streaming pipeline run.
    
    This decorator is specifically designed for functions that return
    a tuple of (Iterator, other_data). It wraps the iterator to keep
    the LangSmith trace context alive during iteration.
    
    Integrates with LangSmith if available and configured.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract question from args (assuming first arg after self is question)
            question = ""
            if len(args) > 1:
                question = str(args[1]) if args[1] else ""
            elif "question" in kwargs:
                question = str(kwargs["question"])
            
            # Generate trace ID
            import uuid
            trace_id = kwargs.get("trace_id") or str(uuid.uuid4())[:8]
            
            # Create pipeline trace
            trace = PipelineTrace(trace_id=trace_id, question=question)
            set_current_trace(trace)
            
            # If LangSmith is available and configured, use trace context manager
            if _is_langsmith_enabled() and langsmith_trace is not None:
                # Create the LangSmith run context - but don't use 'with' statement
                # because we need to keep it alive during iteration
                run_context = langsmith_trace(
                    name=run_name,
                    run_type="chain",
                    inputs={"question": question},
                )
                # Enter the context manually
                run_context.__enter__()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result is a tuple with iterator as first element
                    if isinstance(result, tuple) and len(result) >= 1:
                        iterator, *rest = result
                        if hasattr(iterator, '__iter__') and hasattr(iterator, '__next__'):
                            # Wrap the iterator to keep trace context alive
                            wrapped_iterator = TracedStreamIterator(
                                iterator=iterator,
                                run_context=run_context,
                                trace=trace,
                                step_name="stream_generation",
                            )
                            return (wrapped_iterator, *rest)
                    
                    # If not a streaming result, end the context normally
                    run_context.end(outputs={"result": str(result)[:500] if result else ""})
                    trace.finish()
                    set_current_trace(None)
                    return result
                except Exception as e:
                    run_context.__exit__(type(e), e, e.__traceback__)
                    trace.finish()
                    set_current_trace(None)
                    raise
            else:
                try:
                    return func(*args, **kwargs)
                finally:
                    trace.finish()
                    set_current_trace(None)
        
        return wrapper  # type: ignore
    return decorator
