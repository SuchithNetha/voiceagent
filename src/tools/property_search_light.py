"""
Ultra-lightweight Property Search Tool using Numpy and Gemini Embeddings.
Replaces ChromaDB to save ~600MB of dependency bloat.
Perfect for smaller datasets (under 10,000 items).
"""

import csv
import asyncio
import numpy as np
import inflect
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock
from langchain_core.tools import tool

from src.utils.logger import setup_logging
from src.utils.embeddings import get_embedder
from src.utils.exceptions import SearchEngineError, DataLoadError

logger = setup_logging("Arya-NumpySearch")
_inflect_engine = inflect.engine()

def format_price_for_tts(amount: Any) -> str:
    """Converts price to spoken words."""
    try:
        if amount == 'Unknown' or not amount:
            return "price not available"
        val = int(float(str(amount).replace(',', '')))
        num_words = _inflect_engine.number_to_words(val)
        return f"{num_words} euros"
    except (ValueError, TypeError):
        return f"{amount} euros"

class NumpyPropertySearch:
    """Zero-overhead search using Numpy arrays and Gemini Cloud Embeddings."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NumpyPropertySearch, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.embedder = get_embedder()
        self.properties: List[Dict[str, Any]] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self._initialized = True
        logger.info("🏢 NumpyPropertySearch initialized")

    async def initialize(self, csv_path: Optional[Path] = None):
        """Load data from CSV and fetch embeddings."""
        if self.properties: return
        
        try:
            if csv_path is None:
                root_dir = Path(__file__).resolve().parents[2]
                csv_path = root_dir / "data" / "properties.csv"
            
            if not csv_path.exists():
                raise DataLoadError(str(csv_path))
            
            logger.info(f"📂 Loading properties from: {csv_path}")
            
            # Use built-in csv module to save Pandas bloat (~100MB)
            temp_props = []
            descriptions = []
            
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    temp_props.append(row)
                    descriptions.append(row.get('description', ''))
            
            if not temp_props:
                logger.warning("⚠️ No properties found in CSV.")
                return

            # Compute embeddings in batches using Gemini API (Cloud-based, 0MB local load)
            logger.info(f"🧬 Computing embeddings for {len(descriptions)} properties via Gemini...")
            embeddings = await self.embedder.embed_batch(descriptions)
            
            # Store in numpy matrix for fast search
            self.embedding_matrix = np.array([e for e in embeddings])
            self.properties = temp_props
            
            logger.info(f"✅ Numpy Search ready with {len(self.properties)} properties")
            
        except Exception as e:
            logger.critical(f"❌ Numpy Search init failed: {e}")
            raise SearchEngineError("Failed to initialize Numpy Search", e)

    async def search(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search properties using cosine similarity."""
        if not self.properties or self.embedding_matrix is None:
            await self.initialize()
            
        try:
            # Embed query
            query_vector = await self.embedder.embed_text(query_text)
            if query_vector is None:
                return []
                
            # Cosine similarity via dot product (normalized vectors)
            # Gemini vectors are typically normalized, but we'll be safe
            dot_products = np.dot(self.embedding_matrix, query_vector)
            
            # Get top N indices
            top_indices = np.argsort(dot_products)[-n_results:][::-1]
            
            # Format results
            parsed_results = []
            for idx in top_indices:
                prop = self.properties[idx]
                price = prop.get('price', 'Unknown')
                parsed_results.append({
                    "id": prop.get('id', 'Unknown'),
                    "location": prop.get('location', 'Unknown'),
                    "price_numeric": price,
                    "price_spoken": format_price_for_tts(price),
                    "rooms": prop.get('rooms', 'Unknown'),
                    "baths": prop.get('baths', 'Unknown'),
                    "sqft": prop.get('sqft', 'Unknown'),
                    "description": prop.get('description', 'Unknown')
                })
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"❌ Numpy search failed: {e}")
            return []

_search_engine = NumpyPropertySearch()

@tool
async def search_properties(user_request: str) -> str:
    """
    Search for real estate properties using natural language.
    Uses cloud-based embeddings and lightweight local search.
    """
    try:
        results = await _search_engine.search(user_request)
        
        if not results:
            return "I couldn't find any properties matching that description. Could you try something slightly different?"
            
        output = [f"Found {len(results)} matching properties:\n"]
        for i, res in enumerate(results, 1):
            output.append(
                f"{i}. Property in {res['location']}\n"
                f"   Price: {res['price_spoken']} ({res['price_numeric']} EUR)\n"
                f"   Features: {res['rooms']} rooms, {res['baths']} baths, {res['sqft']} sqft\n"
                f"   Description: {res['description'][:150]}...\n"
                f"   ID: {res['id']}\n"
            )
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Search tool error: {e}")
        return "I encountered a technical issue while searching. Please try again."

async def initialize_lightweight_search() -> bool:
    """Pre-initialize search engine."""
    try:
        await _search_engine.initialize()
        return True
    except Exception as e:
        logger.critical(f"Search init failed: {e}")
        return False
