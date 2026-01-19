"""
STT (Speech-to-Text) Model Loader for Sarah Voice Agent.
"""

from fastrtc import get_stt_model
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError

logger = setup_logging("Models-STT")

def load_stt_model():
    """
    Initialize and return the Speech-to-Text model.
    """
    logger.info("🎤 Loading Speech-to-Text model (Whisper)...")
    try:
        model = get_stt_model()
        logger.info("✅ STT model loaded successfully")
        return model
    except Exception as e:
        logger.critical(f"❌ Failed to load STT model: {e}")
        raise ModelLoadError("Whisper (fastrtc)", original_error=e)
