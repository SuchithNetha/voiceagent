"""
Sarah Voice Agent Utilities Package.

Provides:
- logger: Production-ready logging with rotation and JSON support
- exceptions: Custom exception classes for graceful error handling
"""

from src.utils.logger import setup_logging, get_logger_with_context
from src.utils.exceptions import (
    SarahBaseException,
    ConfigurationError,
    ModelLoadError,
    TranscriptionError,
    SynthesisError,
    SearchEngineError,
    DataLoadError,
    AgentError,
    ToolExecutionError,
    APIError,
    RateLimitError,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger_with_context",
    # Exceptions
    "SarahBaseException",
    "ConfigurationError",
    "ModelLoadError",
    "TranscriptionError",
    "SynthesisError",
    "SearchEngineError",
    "DataLoadError",
    "AgentError",
    "ToolExecutionError",
    "APIError",
    "RateLimitError",
]
