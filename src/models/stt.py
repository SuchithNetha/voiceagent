from fastrtc import get_stt_model
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError
from src.models.stt_groq import load_groq_stt_model

logger = setup_logging("Models-STT")

def load_stt_model():
    """
    Initialize and return the Speech-to-Text model.
    Prefers Groq STT for low latency, falls back to local Whisper.
    """
    # 1. Try Groq STT first (Production preference)
    logger.info("🎤 Attempting to load Groq STT (fast API)...")
    groq_model = load_groq_stt_model()
    if groq_model:
        logger.info("✅ Groq STT loaded successfully")
        return groq_model
    
    # 2. Fallback to local FastRTC Whisper
    logger.info("🎤 Falling back to local Speech-to-Text model (Whisper)...")
    try:
        model = get_stt_model()
        logger.info("✅ Local STT model loaded successfully")
        return model
    except Exception as e:
        logger.critical(f"❌ Failed to load STT model: {e}")
        raise ModelLoadError("Whisper (fastrtc)", original_error=e)
