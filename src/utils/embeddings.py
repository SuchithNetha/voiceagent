"""
Lightweight Cloud Embeddings using Google Gemini API.
Used for property search without heavy local models like Torch.
"""

import os
import asyncio
import numpy as np
import google.generativeai as genai
import warnings
# Suppress the deprecation warning for google-generativeai globally
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

from typing import List, Optional, Union
from src.utils.logger import setup_logging
from src.config import get_config

logger = setup_logging("Utils-Embeddings")
app_config = get_config()

# Initialize Gemini
if os.getenv("GEM_API_KEY"):
    genai.configure(api_key=os.getenv("GEM_API_KEY"))
else:
    logger.warning("GEM_API_KEY not found in environment. Embeddings will fail.")

class GeminiEmbedder:
    """Wrapper for Gemini Embedding API."""
    
    def __init__(self, model: str = "models/embedding-001"):
        self.model = model
        logger.info(f"🧬 Gemini Embedder initialized with {model}")

    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single string."""
        if not text:
            return None
            
        try:
            # Run in thread pool as genai is synchronous
            response = await asyncio.to_thread(
                genai.embed_content,
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return np.array(response['embedding'])
        except Exception as e:
            logger.error(f"❌ Gemini Embedding error: {e}")
            return None

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a list of strings."""
        if not texts:
            return []
            
        try:
            # Gemini supports batching but limited size
            # For simplicity, we'll do everything in one go if small, else batch it
            batch_size = 50
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.model,
                    content=batch,
                    task_type="retrieval_document"
                )
                results.extend([np.array(e) for e in response['embedding']])
                
            return results
        except Exception as e:
            logger.error(f"❌ Gemini Batch Embedding error: {e}")
            return []

def get_embedder():
    """Get the singleton embedder instance."""
    return GeminiEmbedder()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return 0.0
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))
