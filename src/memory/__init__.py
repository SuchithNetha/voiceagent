"""
Memory System for Sarah Voice Agent.

This package provides:
- Session memory for current conversation
- Memory summarization for context management
- Long-term user profile storage
- Redis-based persistent storage
- Cross-session user recognition
"""

from src.memory.models import (
    ConversationTurn,
    SessionMemory,
    UserPreferences,
    PropertyInterest,
    ConversationSnapshot,
    IntentType
)
from src.memory.summarizer import MemorySummarizer
from src.memory.context import ContextWindowManager

# Lazy imports for optional Redis dependency
def get_redis_store():
    """Get Redis store (lazy import)."""
    from src.memory.redis_store import get_redis_store as _get_redis_store
    return _get_redis_store()

def get_session_manager():
    """Get session manager (lazy import)."""
    from src.memory.session_manager import get_session_manager as _get_session_manager
    return _get_session_manager()

__all__ = [
    # Core models
    "ConversationTurn",
    "SessionMemory", 
    "UserPreferences",
    "PropertyInterest",
    "ConversationSnapshot",
    "IntentType",
    # Services
    "MemorySummarizer",
    "ContextWindowManager",
    # Persistent storage (lazy)
    "get_redis_store",
    "get_session_manager",
]

