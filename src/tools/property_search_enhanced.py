"""
Enhanced Property Search Tool for Sarah Voice Agent.

This module extends the base property search with:
- Weighted subjective descriptors (charm, modern, luxury, cozy)
- Soft constraints (preferences, not filters)
- Multi-result ranking
- Context-aware search

Designed for production with:
- Async handling for non-blocking operations
- Graceful error recovery
- Comprehensive logging
"""

import os
import asyncio
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from threading import Lock

import pandas as pd
import numpy as np
import superlinked.framework as sl
from langchain_core.tools import tool
import inflect

from src.utils.logger import setup_logging
from src.utils.exceptions import SearchEngineError, DataLoadError

logger = setup_logging("Sarah-EnhancedSearch")
_inflect_engine = inflect.engine()


# --- SUBJECTIVE DESCRIPTOR KEYWORDS ---
STYLE_KEYWORDS = {
    "charm": [
        "charming", "cozy", "warm", "character", "quaint", "classic", 
        "traditional", "historic", "original", "fireplace", "wooden",
        "period", "elegant", "stately", "refined", "intimate"
    ],
    "modern": [
        "modern", "contemporary", "renovated", "new", "designer",
        "minimalist", "open", "sleek", "smart", "innovative",
        "recent", "updated", "fresh", "current", "state-of-the-art"
    ],
    "luxury": [
        "luxury", "luxurious", "exclusive", "premium", "high-end",
        "prestigious", "upscale", "marble", "jacuzzi", "pool",
        "concierge", "doorman", "penthouse", "spa", "gym"
    ],
    "spacious": [
        "spacious", "large", "big", "ample", "generous", "roomy",
        "expansive", "vast", "open-plan", "double", "oversized"
    ],
    "bright": [
        "bright", "light", "sunny", "luminous", "airy", "windows",
        "balcony", "terrace", "exterior", "south-facing", "east-facing"
    ]
}


def compute_style_score(description: str, style: str) -> float:
    """
    Compute a 0-1 score for how well a description matches a style.
    
    Args:
        description: Property description text
        style: Style category (charm, modern, luxury, etc.)
        
    Returns:
        Score between 0 and 1
    """
    if style not in STYLE_KEYWORDS:
        return 0.0
    
    description_lower = description.lower()
    keywords = STYLE_KEYWORDS[style]
    
    # Count keyword matches with diminishing returns
    matches = sum(1 for kw in keywords if kw in description_lower)
    # Normalize: 3+ matches = 1.0, scaled below
    score = min(1.0, matches / 3)
    
    return round(score, 2)


def enrich_property_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add subjective descriptor columns to property data.
    
    Args:
        df: Property DataFrame with 'description' column
        
    Returns:
        DataFrame with added style score columns
    """
    logger.info("🔧 Enriching property data with style scores...")
    
    for style in STYLE_KEYWORDS.keys():
        col_name = f"{style}_score"
        df[col_name] = df['description'].apply(
            lambda desc: compute_style_score(str(desc), style)
        )
        avg_score = df[col_name].mean()
        logger.debug(f"  {style}: avg={avg_score:.2f}")
    
    logger.info(f"✅ Enriched {len(df)} properties with style scores")
    return df


# --- ENHANCED SUPERLINKED SCHEMA ---
class EnhancedProperty(sl.Schema):
    """Schema with subjective descriptors for weighted search."""
    id: sl.IdField
    description: sl.String
    baths: sl.Float
    rooms: sl.Integer
    sqft: sl.Float
    location: sl.String
    price: sl.Float
    # Subjective style scores (computed from description)
    charm_score: sl.Float
    modern_score: sl.Float
    luxury_score: sl.Float
    spacious_score: sl.Float
    bright_score: sl.Float


@dataclass
class SearchWeights:
    """
    Weights for multi-space search.
    
    These are soft constraints - they influence ranking, not filtering.
    Users can express preferences without hard cutoffs.
    """
    description: float = 0.40    # Natural language match
    charm: float = 0.10          # Subjective: charming/cozy
    modern: float = 0.10         # Subjective: modern/contemporary
    luxury: float = 0.10         # Subjective: luxury/premium
    spacious: float = 0.05       # Subjective: spacious
    bright: float = 0.05         # Subjective: bright/sunny
    price: float = 0.20          # Price preference (soft, not filter)
    
    def normalize(self) -> "SearchWeights":
        """Ensure all weights sum to 1.0."""
        total = (
            self.description + self.charm + self.modern + 
            self.luxury + self.spacious + self.bright + self.price
        )
        if total == 0:
            return SearchWeights()
        
        return SearchWeights(
            description=self.description / total,
            charm=self.charm / total,
            modern=self.modern / total,
            luxury=self.luxury / total,
            spacious=self.spacious / total,
            bright=self.bright / total,
            price=self.price / total
        )
    
    @classmethod
    def from_user_preferences(
        cls, 
        style_preferences: Dict[str, float],
        price_important: bool = False
    ) -> "SearchWeights":
        """Create weights from user preference dict."""
        weights = cls()
        
        # Boost styles user cares about
        for style, strength in style_preferences.items():
            if style == "charm" and hasattr(weights, "charm"):
                weights.charm *= (1 + strength)
            elif style == "modern" and hasattr(weights, "modern"):
                weights.modern *= (1 + strength)
            elif style in ["luxury", "luxurious"]:
                weights.luxury *= (1 + strength)
            elif style in ["spacious", "large"]:
                weights.spacious *= (1 + strength)
            elif style in ["bright", "sunny"]:
                weights.bright *= (1 + strength)
        
        if price_important:
            weights.price *= 1.5
        
        return weights.normalize()


@dataclass 
class SearchEngineState:
    """Encapsulates the Superlinked search engine state."""
    app: Optional[Any] = None
    query: Optional[Any] = None
    is_initialized: bool = False
    initialization_error: Optional[Exception] = None
    property_count: int = 0


class EnhancedPropertySearchEngine:
    """
    Enhanced search engine with weighted multi-space ranking.
    
    Features:
    - Subjective descriptor matching (charm, modern, luxury)
    - Soft price constraints (preferences, not filters)
    - Multi-result ranking
    - Context-aware weight adjustment
    """
    
    def __init__(self):
        self._state = SearchEngineState()
        self._lock = Lock()
        self._property_schema = EnhancedProperty()
        
        # Define core semantic space
        self._description_space = sl.TextSimilaritySpace(
            text=self._property_schema.description,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Core Index - We only index text for Superlinked. 
        # Subjective styles (charm, modern, etc.) are handled by our Hybrid Ranking layer.
        self._property_index = sl.Index(spaces=[self._description_space])
    
    def initialize(self, csv_path: Optional[Path] = None) -> None:
        """
        Initialize the search engine with enriched property data.
        """
        with self._lock:
            if self._state.is_initialized:
                return
            
            if self._state.initialization_error:
                raise SearchEngineError(
                    "Search engine previously failed to initialize",
                    self._state.initialization_error
                )
            
            logger.info("🚀 Initializing Enhanced Search Engine (Hybrid Mode)...")
            
            try:
                # Determine CSV path
                if csv_path is None:
                    root_dir = Path(__file__).resolve().parents[2]
                    csv_path = root_dir / "data" / "properties.csv"
                
                if not csv_path.exists():
                    raise DataLoadError(str(csv_path))
                
                # Load and enrich data
                logger.info(f"📂 Loading properties from: {csv_path}")
                df = pd.read_csv(csv_path, encoding='utf-8')
                df['id'] = df['id'].astype(str)
                
                # Add style scores
                df = enrich_property_data(df)
                
                # Ensure all required columns exist
                for col in ['charm_score', 'modern_score', 'luxury_score', 
                           'spacious_score', 'bright_score']:
                    if col not in df.columns:
                        df[col] = 0.5  # Default neutral score
                
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
                source.put([df])
                
                # Build simpler semantic query
                self._state.query = (
                    sl.Query(self._property_index)
                    .find(self._property_schema)
                    .similar(
                        self._description_space, 
                        sl.Param("description_query"),
                        weight=1.0
                    )
                    .select_all()
                    .limit(sl.Param("result_limit"))
                )
                
                self._state.property_count = len(df)
                self._state.is_initialized = True
                logger.info(f"✅ Hybrid search ready with {len(df)} properties")
                
            except DataLoadError:
                raise
            except Exception as e:
                self._state.initialization_error = e
                logger.critical(f"❌ Enhanced search init failed: {e}", exc_info=True)
                raise SearchEngineError("Failed to initialize enhanced search", e)
    
    def search(
        self,
        query_text: str,
        weights: Optional[SearchWeights] = None,
        price_hint: Optional[float] = None,
        result_limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid ranking: Semantic search + Python-based style scoring.
        """
        if not self._state.is_initialized:
            self.initialize()
        
        weights = weights or SearchWeights()
        norm_weights = weights.normalize()
        
        # Detect style intent from query
        detected_styles = self._detect_styles_in_query(query_text)
        
        try:
            # 1. Fetch more results than needed for re-ranking
            results = self._state.app.query(
                self._state.query,
                description_query=query_text,
                result_limit=max(10, result_limit * 3)
            )
            
            pdf = sl.PandasConverter.to_pandas(results)
            
            if pdf.empty:
                logger.info(f"🔍 No results for: '{query_text}'")
                return []
            
            # 2. Re-rank results in Python using style scores
            properties = []
            for _, row in pdf.iterrows():
                # Base semantic score (from Superlinked)
                semantic_score = row.get('description_space_description_score', 0.5)
                
                # Calculate style boost
                style_score = 0.0
                style_score += row.get('charm_score', 0.5) * norm_weights.charm
                style_score += row.get('modern_score', 0.5) * norm_weights.modern
                style_score += row.get('luxury_score', 0.5) * norm_weights.luxury
                style_score += row.get('spacious_score', 0.5) * norm_weights.spacious
                style_score += row.get('bright_score', 0.5) * norm_weights.bright
                
                # Price score (Gaussian-like for price_hint)
                current_price = row.get('price', 500000)
                diff = abs(current_price - (price_hint or 500000))
                price_score = 1.0 / (1.0 + (diff / 200000))  # Decays by 200k
                
                # Final combined score
                final_score = (
                    (semantic_score * 0.5) + 
                    (style_score * 0.3) + 
                    (price_score * 0.2)
                )
                
                properties.append({
                    "id": row.get('id', 'Unknown'),
                    "location": row.get('location', 'Unknown'),
                    "price_numeric": current_price,
                    "price_spoken": self._format_price_for_tts(current_price),
                    "rooms": row.get('rooms', 'Unknown'),
                    "baths": row.get('baths', 'Unknown'),
                    "sqft": row.get('sqft', 'Unknown'),
                    "description": row.get('description', ''),
                    "charm_score": row.get('charm_score', 0),
                    "modern_score": row.get('modern_score', 0),
                    "luxury_score": row.get('luxury_score', 0),
                    "final_score": final_score
                })
            
            # Sort by final score and limit
            properties.sort(key=lambda x: x['final_score'], reverse=True)
            top_results = properties[:result_limit]
            
            logger.info(f"✅ Optimized {len(top_results)} properties for: '{query_text[:50]}...'")
            return top_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced search failed: {e}", exc_info=True)
            raise SearchEngineError(f"Search failed: {query_text}", e)
    
    def _detect_styles_in_query(self, query: str) -> List[str]:
        """Detect style preferences from natural language query."""
        query_lower = query.lower()
        detected = []
        
        for style, keywords in STYLE_KEYWORDS.items():
            for kw in keywords[:5]:  # Check main keywords
                if kw in query_lower:
                    detected.append(style)
                    break
        
        return detected
    
    def _format_price_for_tts(self, amount: Any) -> str:
        """Convert price to spoken words for TTS."""
        try:
            if amount == 'Unknown' or pd.isna(amount):
                return "price not available"
            num_words = _inflect_engine.number_to_words(int(float(amount)))
            return f"{num_words} euros"
        except (ValueError, TypeError):
            return f"{amount} euros"
    
    async def async_search(
        self,
        query_text: str,
        weights: Optional[SearchWeights] = None,
        price_hint: Optional[float] = None,
        result_limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Async wrapper for search."""
        return await asyncio.to_thread(
            self.search, query_text, weights, price_hint, result_limit
        )


# --- SINGLETON ---
_enhanced_engine: Optional[EnhancedPropertySearchEngine] = None
_engine_lock = Lock()


def get_enhanced_search_engine() -> EnhancedPropertySearchEngine:
    """Get or create singleton enhanced search engine."""
    global _enhanced_engine
    
    if _enhanced_engine is None:
        with _engine_lock:
            if _enhanced_engine is None:
                _enhanced_engine = EnhancedPropertySearchEngine()
    
    return _enhanced_engine


# --- LANGCHAIN TOOL ---
@tool
async def search_properties_enhanced(
    user_request: str,
    style_preference: Optional[Any] = None,
    price_max: Optional[Any] = None
) -> str:
    """
    Search for real estate properties with style preferences.
    
    Use this when users ask about properties. Supports style preferences
    like 'modern', 'charming', 'luxury', 'spacious', or 'bright'.
    
    Args:
        user_request: The user's natural language query
        style_preference: Optional style or 'none' (modern, charming, luxury, etc.)
        price_max: Optional maximum price or 'none'
        
    Returns:
        Formatted property information
    """
    try:
        # Robust parameter parsing (handle strings like "none", "null")
        actual_style = None
        if style_preference and str(style_preference).lower() not in ["none", "null", "false"]:
            actual_style = str(style_preference)
            
        actual_price = None
        if price_max is not None:
            try:
                price_str = str(price_max).lower()
                if price_str not in ["none", "null", "false"]:
                    # Clean currency symbols and commas
                    price_val = re.sub(r'[^\d.]', '', price_str)
                    if price_val:
                        actual_price = float(price_val)
            except (ValueError, TypeError):
                pass

        engine = get_enhanced_search_engine()
        
        # Build weights from style preference
        weights = SearchWeights()
        if actual_style:
            style_pref = actual_style.lower()
            weights = SearchWeights.from_user_preferences(
                {style_pref: 0.5},
                price_important=(actual_price is not None)
            )
        
        results = await engine.async_search(
            user_request,
            weights=weights,
            price_hint=actual_price,
            result_limit=3
        )
        
        if not results:
            return "No properties found matching your criteria. Would you like to broaden your search?"
        
        # Format for agent
        output_parts = [f"Found {len(results)} matching properties:\n"]
        
        for i, prop in enumerate(results, 1):
            output_parts.append(
                f"\n{i}. Property in {prop['location']}\n"
                f"   Price: {prop['price_spoken']} ({prop['price_numeric']:,.0f} EUR)\n"
                f"   Rooms: {prop['rooms']} | Baths: {prop['baths']} | Size: {prop['sqft']} sqft\n"
                f"   ID: {prop['id']}"
            )
            
            # Note style matches
            styles = []
            if prop.get('charm_score', 0) > 0.6:
                styles.append("charming")
            if prop.get('modern_score', 0) > 0.6:
                styles.append("modern")
            if prop.get('luxury_score', 0) > 0.6:
                styles.append("luxurious")
            if styles:
                output_parts.append(f"   Style: {', '.join(styles)}")
        
        return "\n".join(output_parts)
        
    except SearchEngineError as e:
        logger.error(f"Enhanced search error: {e}")
        return e.user_message
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return "I had trouble searching. Let me try again."


def initialize_enhanced_search_engine() -> bool:
    """Pre-initialize the enhanced search engine."""
    try:
        engine = get_enhanced_search_engine()
        engine.initialize()
        return True
    except Exception as e:
        logger.critical(f"Enhanced search init failed: {e}")
        return False
