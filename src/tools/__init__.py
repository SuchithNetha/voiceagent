"""
Arya Voice Agent Tools Package.

Provides LangChain tools for the agent to use:
- search_properties: Lightweight semantic search using Numpy and Gemini Cloud Embeddings.
"""

from src.tools.property_search_light import (
    search_properties,
    initialize_lightweight_search as initialize_search_engine,
)

def get_available_tools():
    """Return the list of tools available for the agent."""
    return [search_properties]

__all__ = [
    "search_properties",
    "initialize_search_engine",
    "get_available_tools",
]
