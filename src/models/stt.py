from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError
from src.models.stt_groq import load_groq_stt_model

logger = setup_logging("Models-STT")

def load_stt_model():
    """
    Initialize and return the Speech-to-Text model.
    Uses Groq STT for performance and low memory footprint.
    """
    logger.info("🎤 Loading Groq STT (API-based)...")
    groq_model = load_groq_stt_model()
    if groq_model:
        logger.info("✅ Groq STT loaded successfully")
        return groq_model
    
    # We remove the local fallback to save ~500MB of RAM
    logger.critical("❌ GROQ_API_KEY for STT not found and local fallback disabled for memory safety.")
    raise ModelLoadError("STT (Groq)", original_error="GROQ_API_KEY missing")
