"""Debug utilities for RAG pipeline visualization with colored output."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.types import Answer, ContextChunk, Intent, RetrievalPlan


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    
    # Foreground colors
    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"
    
    # Bright foreground
    BRIGHT_RED = "\x1b[91m"
    BRIGHT_GREEN = "\x1b[92m"
    BRIGHT_YELLOW = "\x1b[93m"
    BRIGHT_BLUE = "\x1b[94m"
    BRIGHT_MAGENTA = "\x1b[95m"
    BRIGHT_CYAN = "\x1b[96m"
    
    # Background colors
    BG_BLACK = "\x1b[40m"
    BG_RED = "\x1b[41m"
    BG_GREEN = "\x1b[42m"
    BG_YELLOW = "\x1b[43m"
    BG_BLUE = "\x1b[44m"
    BG_MAGENTA = "\x1b[45m"
    BG_CYAN = "\x1b[46m"


# Pipeline stage colors
STAGE_COLORS = {
    "intent": Colors.BRIGHT_CYAN,
    "router": Colors.BRIGHT_YELLOW,
    "local": Colors.BRIGHT_GREEN,
    "web": Colors.BRIGHT_BLUE,
    "fusion": Colors.BRIGHT_MAGENTA,
    "generator": Colors.BRIGHT_RED,
}


def _colored(text: str, color: str) -> str:
    """Wrap text with color codes."""
    return f"{color}{text}{Colors.RESET}"


def _bold(text: str) -> str:
    """Make text bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def _dim(text: str) -> str:
    """Make text dim."""
    return f"{Colors.DIM}{text}{Colors.RESET}"


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


class DebugPrinter:
    """Debug printer for RAG pipeline stages with colored output."""
    
    def __init__(self, enabled: bool = False, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self._step_timings: Dict[str, float] = {}  # Store step durations
        self._pipeline_start: Optional[float] = None
    
    def _print(self, *args, **kwargs):
        """Print to configured stream."""
        if self.enabled:
            print(*args, file=self.stream, **kwargs)
    
    def _format_duration(self, ms: float) -> str:
        """Format duration in appropriate units."""
        if ms < 1000:
            return f"{ms:.0f}ms"
        else:
            return f"{ms/1000:.2f}s"
    
    def _header(self, stage: str, title: str, duration_ms: Optional[float] = None):
        """Print stage header with colored box and optional timing."""
        color = STAGE_COLORS.get(stage, Colors.WHITE)
        width = 60
        border = "─" * width
        self._print()
        self._print(_colored(f"┌{border}┐", color))
        
        # Add timing to header if provided
        if duration_ms is not None:
            duration_str = self._format_duration(duration_ms)
            timing_display = _dim(f" ⏱ {duration_str}")
            self._print(_colored(f"│ {_bold(title)}{timing_display:<{width-len(title)-1}}│", color))
        else:
            self._print(_colored(f"│ {_bold(title):<{width-1}}│", color))
        
        self._print(_colored(f"└{border}┘", color))
    
    def _kv(self, key: str, value: Any, indent: int = 2, color: str = ""):
        """Print key-value pair."""
        prefix = " " * indent
        if color:
            self._print(f"{prefix}{_colored(key + ':', color)} {value}")
        else:
            self._print(f"{prefix}{_dim(key + ':')} {value}")
    
    def _list_item(self, item: str, indent: int = 4, bullet: str = "•"):
        """Print list item."""
        prefix = " " * indent
        self._print(f"{prefix}{_dim(bullet)} {item}")
    
    def start_pipeline(self):
        """Mark the start of the pipeline for timing."""
        self._pipeline_start = time.perf_counter()
        self._step_timings = {}
    
    def record_step_time(self, step_name: str, duration_ms: float):
        """Record the duration of a step."""
        self._step_timings[step_name] = duration_ms
    
    def print_question(self, question: str):
        """Print the input question."""
        if not self.enabled:
            return
        self.start_pipeline()  # Start timing when question is received
        self._header("intent", "📝 INPUT QUESTION")
        self._print(f"  {_bold(question)}")
    
    def print_intent(self, intent: Intent, confidence: float, rationales: List[str] = None, duration_ms: Optional[float] = None):
        """Print intent classification result."""
        if not self.enabled:
            return
        color = STAGE_COLORS["intent"]
        self._header("intent", "🧠 INTENT CLASSIFICATION", duration_ms)
        if duration_ms:
            self.record_step_time("intent_classification", duration_ms)
        
        # Intent with color-coded confidence
        conf_color = Colors.GREEN if confidence >= 0.8 else (Colors.YELLOW if confidence >= 0.6 else Colors.RED)
        self._kv("Intent", _colored(intent.value, color), color=color)
        self._kv("Confidence", _colored(f"{confidence:.2%}", conf_color))
        
        if rationales:
            self._print(f"  {_dim('Rationales:')}")
            for r in rationales:
                self._list_item(r)
    
    def print_routing(self, plan: RetrievalPlan, intent: Intent):
        """Print retrieval routing decision."""
        if not self.enabled:
            return
        color = STAGE_COLORS["router"]
        self._header("router", "🔀 RETRIEVAL ROUTING")
        
        self._kv("Local Retrieval", f"top_k={plan.local_top_k}")
        self._kv("Strategy", plan.hybrid_strategy)
        self._kv("Based on Intent", _colored(intent.value, color))
    
    def print_local_retrieval(self, chunks: List[ContextChunk], query: str, duration_ms: Optional[float] = None):
        """Print local retrieval results summary."""
        if not self.enabled:
            return
        color = STAGE_COLORS["local"]
        self._header("local", f"📚 LOCAL RETRIEVAL ({len(chunks)} chunks)", duration_ms)
        if duration_ms:
            self.record_step_time("local_retrieval", duration_ms)
        
        if not chunks:
            self._print(f"  {_dim('No chunks retrieved')}")
            return
        
        # 只输出汇总信息，不输出每条结果的详情
        # 统计各类型来源数量
        sql_count = sum(1 for ch in chunks if ch.source_id == "sql_database")
        table_count = sum(1 for ch in chunks if ch.metadata.get("doctype") == "table" and ch.source_id != "sql_database")
        text_count = len(chunks) - sql_count - table_count
        
        # 计算相似度统计
        scores = [ch.similarity for ch in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        self._kv("SQL 结果", f"{sql_count} 条")
        self._kv("表格数据", f"{table_count} 条")
        self._kv("文本数据", f"{text_count} 条")
        self._kv("相似度", f"avg={avg_score:.3f}, max={max_score:.3f}, min={min_score:.3f}")
    
    def print_fusion(self, chunks: List[ContextChunk], input_count: int, duration_ms: Optional[float] = None):
        """Print fusion/retrieval results."""
        if not self.enabled:
            return
        color = STAGE_COLORS["fusion"]
        self._header("fusion", f"⚗️  FUSION LAYER ({len(chunks)} selected)", duration_ms)
        if duration_ms:
            self.record_step_time("fusion", duration_ms)
        
        self._kv("Input", f"{input_count} chunks")
        self._kv("Output", f"{len(chunks)} chunks after fusion")
        
        if not chunks:
            self._print(f"  {_dim('No chunks after fusion')}")
            return
        
        # Score statistics
        scores = [ch.similarity for ch in chunks]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            self._print()
            self._kv("Score Stats", f"avg={avg_score:.3f}, max={max_score:.3f}, min={min_score:.3f}")
        
        self._print()
        self._print(f"  {_dim('Ranked chunks:')}")
        for i, ch in enumerate(chunks, 1):
            score = ch.similarity
            src_icon = "📚" if ch.source_type == "local" else "🌐"
            score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            title = _truncate(ch.title or ch.source_id or "untitled", 40)
            self._print(f"    {_colored(f'{i}.', color)} {src_icon} [{score_bar}] {score:.3f} | {title}")
    
    def print_generation(self, answer: Answer, context_count: int, duration_ms: Optional[float] = None):
        """Print generation result."""
        if not self.enabled:
            return
        color = STAGE_COLORS["generator"]
        self._header("generator", "✨ ANSWER GENERATION", duration_ms)
        if duration_ms:
            self.record_step_time("generation", duration_ms)
        
        # Confidence with color coding
        conf = answer.confidence
        conf_color = Colors.GREEN if conf >= 0.7 else (Colors.YELLOW if conf >= 0.5 else Colors.RED)
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        
        self._kv("Generator", answer.meta.get("generator", "unknown"))
        self._kv("Context Chunks", context_count)
        self._kv("Confidence", f"{_colored(f'[{conf_bar}]', conf_color)} {conf:.2%}")
        
        self._print()
        self._print(f"  {_dim('Citations')} ({len(answer.citations)}):")
        for c in answer.citations[:5]:  # Show first 5
            self._list_item(_truncate(str(c), 70))
        if len(answer.citations) > 5:
            self._print(f"      {_dim(f'... and {len(answer.citations) - 5} more')}")
        
        self._print()
        self._print(f"  {_dim('Answer Preview:')}")
        preview = _truncate(answer.text, 200)
        self._print(f"    {preview}")
    
    def print_summary(self, answer: Answer, meta: Dict[str, Any]):
        """Print final summary with timing breakdown."""
        if not self.enabled:
            return
        
        # Calculate total time
        total_ms = 0.0
        if self._pipeline_start:
            total_ms = (time.perf_counter() - self._pipeline_start) * 1000
        
        self._print()
        self._print(_colored("=" * 64, Colors.DIM))
        self._print(_bold("📊 PIPELINE SUMMARY"))
        self._print(_colored("=" * 64, Colors.DIM))
        
        self._kv("Intent", meta.get("intent", "N/A"))
        self._kv("Intent Confidence", meta.get("intent_confidence", "N/A"))
        self._kv("Chunks Retrieved", meta.get("chunks_retrieved", meta.get("local_chunks", "0")))
        self._kv("Final Confidence", f"{answer.confidence:.2%}")
        self._kv("Citations Count", len(answer.citations))
        self._kv("Answer Length", f"{len(answer.text)} chars")
        
        # Print timing breakdown
        if self._step_timings:
            self._print()
            self._print(f"  {_bold('⏱ Timing Breakdown:')}")
            for step_name, duration in self._step_timings.items():
                bar_len = min(int(duration / 100), 30)  # Scale: 100ms = 1 char, max 30 chars
                bar = "█" * bar_len
                step_display = step_name.replace("_", " ").title()
                self._print(f"    {step_display:<20} {_colored(bar, Colors.CYAN)} {self._format_duration(duration)}")
            
            if total_ms > 0:
                self._print(f"    {'─' * 40}")
                self._print(f"    {'Total':<20} {_bold(self._format_duration(total_ms))}")
        
        self._print()


# Global debug printer instance
_debug_printer: Optional[DebugPrinter] = None


def get_debug_printer() -> DebugPrinter:
    """Get the global debug printer instance."""
    global _debug_printer
    if _debug_printer is None:
        _debug_printer = DebugPrinter(enabled=False)
    return _debug_printer


def set_debug_mode(enabled: bool):
    """Enable or disable debug mode globally."""
    global _debug_printer
    _debug_printer = DebugPrinter(enabled=enabled)


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return get_debug_printer().enabled


def log_retrieval_step(stage: str, results: List[Any], query: str = "", log_file: str = "retrieval.log"):
    """Log retrieval results to a file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{timestamp}] Stage: {stage}\n")
            if query:
                f.write(f"Query: {query}\n")
            f.write(f"Result Count: {len(results)}\n")
            f.write(f"{'-'*80}\n")
            
            for i, item in enumerate(results, 1):
                f.write(f"Item {i}:\n")
                # Handle SearchResult or ContextChunk or other objects
                if hasattr(item, 'id'):
                    f.write(f"  ID: {item.id}\n")
                
                # Score/Similarity
                if hasattr(item, 'score'):
                    f.write(f"  Score: {item.score}\n")
                elif hasattr(item, 'similarity'):
                    f.write(f"  Similarity: {item.similarity}\n")
                
                # Source
                if hasattr(item, 'source'):
                    f.write(f"  Source: {item.source}\n")
                elif hasattr(item, 'source_id'):
                    f.write(f"  Source ID: {item.source_id}\n")
                
                # Content
                if hasattr(item, 'content'):
                    content = str(item.content).replace('\n', ' ')
                    if len(content) > 500:
                        content = content[:500] + "..."
                    f.write(f"  Content: {content}\n")
                
                # Metadata
                if hasattr(item, 'metadata'):
                    f.write(f"  Metadata: {item.metadata}\n")
                f.write("\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}", file=sys.stderr)

