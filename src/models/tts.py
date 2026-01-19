"""
TTS (Text-to-Speech) Model Loader for Sarah Voice Agent.
"""

from fastrtc import get_tts_model
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError

logger = setup_logging("Models-TTS")

def load_tts_model():
    """
    Initialize and return the Text-to-Speech model.
    """
    logger.info("🔊 Loading Text-to-Speech model (Kokoro)...")
    try:
        model = get_tts_model()
        logger.info("✅ TTS model loaded successfully")
        return model
    except Exception as e:
        logger.critical(f"❌ Failed to load TTS model: {e}")
        raise ModelLoadError("Kokoro (fastrtc)", original_error=e)
