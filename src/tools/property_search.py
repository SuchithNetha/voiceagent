"""
Property Search Tool for Sarah Voice Agent.

This module provides semantic search over real estate properties using Superlinked.
It's designed to be production-ready with:
- Proper async handling for non-blocking operations
- Custom exception handling for graceful error recovery
- Thread-safe initialization (no global mutable state)
- Comprehensive logging for debugging
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock

import pandas as pd
import superlinked.framework as sl
from langchain_core.tools import tool
import inflect

from src.utils.logger import setup_logging
from src.utils.exceptions import (
    SearchEngineError,
    DataLoadError,
)

# --- INITIALIZE LOGGING & UTILS ---
logger = setup_logging("Sarah-Search")
_inflect_engine = inflect.engine()


def format_price_for_tts(amount: Any) -> str:
    """
    Converts numeric price to spoken words for natural TTS output.
    
    Args:
        amount: The price value (can be int, float, or 'Unknown')
        
    Returns:
        Human-readable price string (e.g., "five hundred thousand euros")
    """
    try:
        if amount == 'Unknown' or pd.isna(amount):
            return "price not available"
        num_words = _inflect_engine.number_to_words(int(float(amount)))
        return f"{num_words} euros"
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not format price '{amount}': {e}")
        return f"{amount} euros"


# --- SUPERLINKED SCHEMA DEFINITION ---
class Property(sl.Schema):
    """Schema definition for property data in Superlinked."""
    id: sl.IdField
    description: sl.String
    baths: sl.Float
    rooms: sl.Integer
    sqft: sl.Float
    location: sl.String
    price: sl.Float


@dataclass
class SearchEngineState:
    """
    Encapsulates the Superlinked search engine state.
    
    Using a dataclass instead of global variables provides:
    - Thread safety with explicit locking
    - Clear state management
    - Easier testing (can create fresh instances)
    """
    app: Optional[Any] = None
    query: Optional[Any] = None
    is_initialized: bool = False
    initialization_error: Optional[Exception] = None


class PropertySearchEngine:
    """
    Thread-safe wrapper for the Superlinked search engine.
    
    This class manages the lifecycle of the search engine and provides
    thread-safe access for concurrent requests.
    """
    
    def __init__(self):
        self._state = SearchEngineState()
        self._lock = Lock()
        self._property_schema = Property()
        
        # Define embedding spaces
        self._description_space = sl.TextSimilaritySpace(
            text=self._property_schema.description,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._price_space = sl.NumberSpace(
            number=self._property_schema.price,
            min_value=50000,
            max_value=20000000,
            mode=sl.Mode.MINIMUM
        )
        self._property_index = sl.Index(
            spaces=[self._description_space, self._price_space]
        )
    
    def initialize(self, csv_path: Optional[Path] = None) -> None:
        """
        Initialize the search engine with property data.
        
        This is thread-safe and will only initialize once.
        Subsequent calls are no-ops if already initialized.
        
        Args:
            csv_path: Path to the properties CSV file
            
        Raises:
            DataLoadError: If CSV file cannot be loaded
            SearchEngineError: If Superlinked initialization fails
        """
        with self._lock:
            if self._state.is_initialized:
                return
            
            if self._state.initialization_error:
                # Re-raise previous error if initialization failed before
                raise SearchEngineError(
                    "Search engine previously failed to initialize",
                    self._state.initialization_error
                )
            
            logger.info("🚀 Initializing Sarah's search engine...")
            
            try:
                # Determine CSV path
                if csv_path is None:
                    root_dir = Path(__file__).resolve().parents[2]
                    csv_path = root_dir / "data" / "properties.csv"
                
                # Validate CSV exists
                if not csv_path.exists():
                    raise DataLoadError(str(csv_path))
                
                # Create source and executor
                source = sl.InMemorySource(
                    self._property_schema,
                    parser=sl.DataFrameParser(schema=self._property_schema)
                )
                executor = sl.InMemoryExecutor(
                    sources=[source],
                    indices=[self._property_index]
                )
                self._state.app = executor.run()
                
                # Load property data
                logger.info(f"📂 Loading properties from: {csv_path}")
                df = pd.read_csv(csv_path, encoding='utf-8')
                df['id'] = df['id'].astype(str)
                source.put([df])
                
                # Pre-build query template
                self._state.query = (
                    sl.Query(self._property_index)
                    .find(self._property_schema)
                    .similar(self._description_space, sl.Param("n_query"))
                    .select_all()
                    .limit(1)
                )
                
                self._state.is_initialized = True
                logger.info(f"✅ Search engine ready with {len(df)} properties")
                
            except DataLoadError:
                raise
            except Exception as e:
                self._state.initialization_error = e
                logger.critical(f"❌ Search engine initialization failed: {e}", exc_info=True)
                raise SearchEngineError("Failed to initialize search engine", e)
    
    def search(self, query_text: str) -> Dict[str, Any]:
        """
        Perform a synchronous search (to be run in thread pool).
        
        Args:
            query_text: Natural language search query
            
        Returns:
            Dictionary with property details or error information
            
        Raises:
            SearchEngineError: If search fails
        """
        if not self._state.is_initialized:
            self.initialize()
        
        try:
            results = self._state.app.query(
                self._state.query,
                n_query=query_text
            )
            
            pdf = sl.PandasConverter.to_pandas(results)
            
            if pdf.empty:
                logger.info(f"🔍 No results for query: '{query_text}'")
                return {
                    "status": "No Results",
                    "message": "No properties found matching your criteria."
                }
            
            # Extract top result
            res = pdf.iloc[0]
            raw_price = res.get('price', 'Unknown')
            
            property_data = {
                "status": "Success",
                "id": res.get('id', 'Unknown'),
                "location": res.get('location', 'Unknown'),
                "price_numeric": raw_price,
                "price_spoken": format_price_for_tts(raw_price),
                "rooms": res.get('rooms', 'Unknown'),
                "baths": res.get('baths', 'Unknown'),
                "sqft": res.get('sqft', 'Unknown'),
                "description": res.get('description', 'No description available'),
            }
            
            logger.info(f"✅ Found property in {property_data['location']}")
            return property_data
            
        except Exception as e:
            logger.error(f"❌ Search failed for '{query_text}': {e}", exc_info=True)
            raise SearchEngineError(f"Search failed: {query_text}", e)
    
    async def async_search(self, query_text: str) -> Dict[str, Any]:
        """
        Perform an async search (runs blocking search in thread pool).
        
        This is the preferred method for use in async contexts like
        the voice handler, as it doesn't block the event loop.
        
        Args:
            query_text: Natural language search query
            
        Returns:
            Dictionary with property details
        """
        logger.info(f"🔍 Searching for: '{query_text}'")
        return await asyncio.to_thread(self.search, query_text)


# --- SINGLETON INSTANCE ---
# We use a module-level instance but it's properly encapsulated
_search_engine: Optional[PropertySearchEngine] = None
_engine_lock = Lock()


def get_search_engine() -> PropertySearchEngine:
    """
    Get or create the singleton search engine instance.
    
    Thread-safe lazy initialization pattern.
    """
    global _search_engine
    
    if _search_engine is None:
        with _engine_lock:
            # Double-check after acquiring lock
            if _search_engine is None:
                _search_engine = PropertySearchEngine()
    
    return _search_engine


# --- LANGCHAIN TOOL ---
@tool
async def search_properties(user_request: str) -> str:
    """
    Search for real estate properties using natural language.
    
    Use this tool when the user asks about properties, apartments, houses,
    or real estate in Madrid. Returns details about the best matching property.
    
    Args:
        user_request: The user's natural language query about properties
        
    Returns:
        A string with property details or an error message
    """
    try:
        engine = get_search_engine()
        result = await engine.async_search(user_request)
        
        if result.get("status") == "No Results":
            return result["message"]
        
        # Format response for the agent
        return (
            f"Property Found:\n"
            f"- Location: {result['location']}\n"
            f"- Price: {result['price_spoken']} ({result['price_numeric']} EUR)\n"
            f"- Rooms: {result['rooms']}\n"
            f"- Bathrooms: {result['baths']}\n"
            f"- Size: {result['sqft']} sqft\n"
            f"- Description: {result['description']}\n"
            f"- Property ID: {result['id']}"
        )
        
    except SearchEngineError as e:
        logger.error(f"Search tool error: {e}")
        return e.user_message
        
    except Exception as e:
        logger.error(f"Unexpected error in search_properties: {e}", exc_info=True)
        return "I encountered a technical issue while searching. Please try again."


# --- INITIALIZATION HOOK ---
def initialize_search_engine() -> bool:
    """
    Pre-initialize the search engine.
    
    Call this at application startup to fail fast if there are issues.
    
    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        engine = get_search_engine()
        engine.initialize()
        return True
    except Exception as e:
        logger.critical(f"Search engine failed to initialize: {e}")
        return False