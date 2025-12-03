"""Logging utilities with trace ID support and colored output."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def set_logging_debug_mode(enabled: bool):
    """Set whether debug mode is enabled for logging (no-op, kept for compatibility)."""
    pass


def is_logging_debug_mode() -> bool:
    """Check if debug mode is enabled (always returns False, kept for compatibility)."""
    return False


class TraceAdapter(logging.LoggerAdapter):
    """Logger adapter that prepends trace ID to messages.
    
    INFO and DEBUG logs are silenced and replaced by colorful debug output.
    WARNING and ERROR logs are always shown.
    """
    
    def __init__(self, logger: logging.Logger, trace_id: Optional[str]):
        super().__init__(logger, {"trace_id": trace_id})

    def process(self, msg, kwargs):
        trace = self.extra.get("trace_id")
        prefix = f"[trace={trace}] " if trace else ""
        return prefix + str(msg), kwargs
    
    def info(self, msg, *args, **kwargs):
        """INFO logs are silenced (replaced by colorful debug output)."""
        pass
    
    def debug(self, msg, *args, **kwargs):
        """DEBUG logs are silenced (replaced by colorful debug output)."""
        pass
    
    def warning(self, msg, *args, **kwargs):
        """Warnings are always shown."""
        super().warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Errors are always shown."""
        super().error(msg, *args, **kwargs)


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log messages."""
    
    COLORS = {
        logging.DEBUG: "\x1b[36m",    # Cyan
        logging.INFO: "\x1b[90m",     # Gray (dimmed)
        logging.WARNING: "\x1b[33m",  # Yellow
        logging.ERROR: "\x1b[31m",    # Red
        logging.CRITICAL: "\x1b[41m", # Red background
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        reset = "\x1b[0m" if color else ""
        return f"{color}{base}{reset}"


def get_logger(name: str, trace_id: Optional[str] = None) -> TraceAdapter:
    """Get a logger with optional trace ID support.
    
    Args:
        name: Logger name (typically __class__.__name__)
        trace_id: Optional trace ID for request tracking
        
    Returns:
        A LoggerAdapter with trace ID support
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        fmt = ColorFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Allow all levels, filtering done in adapter
        # Avoid propagating to root to keep format consistent
        logger.propagate = False
    return TraceAdapter(logger, trace_id)
