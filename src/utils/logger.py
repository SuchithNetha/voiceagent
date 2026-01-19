"""
Production-Ready Logging Module for Sarah Voice Agent.

This module provides:
- Console logging (human-readable)
- File logging with rotation (persistent, won't fill disk)
- JSON formatting option (for log aggregation tools)
- Environment-based configuration
- Request context support (for tracing)

Usage:
    from src.utils.logger import setup_logging
    logger = setup_logging("Sarah-Main")
    logger.info("Application started")
    logger.error("Something failed", exc_info=True)
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for easier parsing by log aggregation tools.
    
    In production, tools like Elasticsearch, Datadog, or CloudWatch 
    can easily parse JSON logs and let you search/filter/alert on them.
    
    Example output:
    {"timestamp": "2026-01-13T12:00:00.000Z", "level": "INFO", "message": "User query received"}
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Include any extra fields passed to the logger
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
            
        return json.dumps(log_data, ensure_ascii=False)


# --- CONFIGURATION ---
# Read from environment variables for flexibility (no code changes needed)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_JSON_FORMAT = os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"

# Path configuration
LOG_DIR = Path(__file__).parent.parent.parent / "logs"  # voiceagent/logs/

# Format patterns
LOG_FORMAT_CONSOLE = "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"
LOG_FORMAT_FILE = "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"

# Rotation settings (prevents disk from filling up)
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 5  # Keep 5 old log files (sarah.log.1, sarah.log.2, etc.)


def setup_logging(name: str, use_json: Optional[bool] = None) -> logging.Logger:
    """
    Sets up a production-ready logger with console and file output.
    
    Features:
    - Console output with human-readable format
    - File output with automatic rotation (prevents disk full)
    - Optional JSON format for log aggregation tools
    - Configurable via environment variables
    
    Args:
        name: The name of the logger (e.g., 'Sarah-Main', 'Sarah-Search')
        use_json: If True, use JSON format for file logs. 
                  If None, reads from LOG_JSON_FORMAT env var.
        
    Returns:
        A configured logger instance
        
    Environment Variables:
        LOG_LEVEL: Set log verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_TO_FILE: Enable/disable file logging ("true" or "false")
        LOG_JSON_FORMAT: Use JSON format for file logs ("true" or "false")
        
    Example:
        logger = setup_logging("Sarah-Main")
        logger.info("Application started")
        logger.warning("Low memory detected")
        logger.error("Database connection failed", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    # Prevent adding duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger
    
    # Set the base log level from environment
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    
    # --- CONSOLE HANDLER (Human-readable) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Console shows all levels
    console_formatter = logging.Formatter(LOG_FORMAT_CONSOLE, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # --- FILE HANDLER (Persistent + Rotating) ---
    if LOG_TO_FILE:
        try:
            # Create logs directory if it doesn't exist
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            
            log_file = LOG_DIR / "sarah.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)  # File captures everything
            
            # Determine if JSON format should be used
            should_use_json = use_json if use_json is not None else LOG_JSON_FORMAT
            
            if should_use_json:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(LOG_FORMAT_FILE, datefmt="%Y-%m-%d %H:%M:%S")
                )
            
            logger.addHandler(file_handler)
            
        except PermissionError:
            logger.warning(f"⚠️  No permission to write logs to {LOG_DIR}")
        except Exception as e:
            logger.warning(f"⚠️  Could not set up file logging: {e}")
    
    return logger


def get_logger_with_context(
    name: str, 
    request_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> logging.LoggerAdapter:
    """
    Creates a logger adapter that includes request context in every log message.
    
    This is useful for tracing a single user's request across all log lines
    in a multi-user production environment.
    
    Args:
        name: The name of the logger
        request_id: Unique identifier for the current request/session
        user_id: Identifier for the current user
        
    Returns:
        A LoggerAdapter that automatically includes context in log messages
        
    Example:
        # In your request handler:
        logger = get_logger_with_context("Sarah-Handler", request_id="req-abc-123")
        logger.info("Processing voice input")  
        # Output: 13:45:00 | INFO | Sarah-Handler | [req-abc-123] Processing voice input
    """
    base_logger = setup_logging(name)
    
    context = {}
    if request_id:
        context["request_id"] = request_id
    if user_id:
        context["user_id"] = user_id
    
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Prepend context to message
            prefix_parts = []
            if self.extra.get("request_id"):
                prefix_parts.append(f"[{self.extra['request_id']}]")
            if self.extra.get("user_id"):
                prefix_parts.append(f"[user:{self.extra['user_id']}]")
            
            prefix = " ".join(prefix_parts)
            if prefix:
                msg = f"{prefix} {msg}"
            
            return msg, kwargs
    
    return ContextAdapter(base_logger, context)
