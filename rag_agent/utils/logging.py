"""Logging utilities with trace ID support and colored output."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Log format
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def set_logging_debug_mode(enabled: bool):
    """Set whether debug mode is enabled for logging."""
    level = logging.DEBUG if enabled else DEFAULT_LOG_LEVEL
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def is_logging_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    root_logger = logging.getLogger()
    return root_logger.level <= logging.DEBUG


class TraceAdapter(logging.LoggerAdapter):
    """Logger adapter that prepends trace ID to messages."""
    
    def __init__(self, logger: logging.Logger, trace_id: Optional[str]):
        super().__init__(logger, {"trace_id": trace_id})

    def process(self, msg, kwargs):
        trace = self.extra.get("trace_id")
        prefix = f"[trace={trace}] " if trace else ""
        return prefix + str(msg), kwargs


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log messages."""
    
    COLORS = {
        logging.DEBUG: "\x1b[36m",    # Cyan
        logging.INFO: "\x1b[32m",     # Green
        logging.WARNING: "\x1b[33m",  # Yellow
        logging.ERROR: "\x1b[31m",    # Red
        logging.CRITICAL: "\x1b[41m", # Red background
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        reset = "\x1b[0m" if color else ""
        return f"{color}{base}{reset}"


def _setup_logging():
    """Set up logging with environment-based configuration."""
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Skip if already configured
    if root_logger.handlers:
        return
    
    # Get log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, DEFAULT_LOG_LEVEL)
    
    # Create handler
    handler = logging.StreamHandler(stream=sys.stderr)
    
    # Create formatter
    formatter = ColorFormatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger.propagate = False


def get_logger(name: str, trace_id: Optional[str] = None) -> TraceAdapter:
    """Get a logger with optional trace ID support.
    
    Args:
        name: Logger name (typically __class__.__name__)
        trace_id: Optional trace ID for request tracking
        
    Returns:
        A LoggerAdapter with trace ID support
    """
    # Ensure logging is set up
    _setup_logging()
    
    # Get logger
    logger = logging.getLogger(name)
    logger.propagate = True
    
    return TraceAdapter(logger, trace_id)
