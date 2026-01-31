import os
from src.utils.logger import setup_logging

logger = setup_logging("Models-Init")

def load_stt_model():
    """Load STT model based on config."""
    stt_type = os.getenv("STT_MODEL", "groq").lower()
    
    if stt_type == "sarvam":
        from src.models.stt_sarvam import load_stt_model as load
        return load()
    else:
        from src.models.stt_groq import load_groq_stt_model
        return load_groq_stt_model()

def load_tts_model():
    """Load TTS model based on config."""
    tts_type = os.getenv("TTS_MODEL", "edge").lower()
    
    if tts_type == "sarvam":
        from src.models.tts_sarvam import load_tts_model as load
        return load()
    else:
        from src.models.tts import load_tts_model as load
        return load()

from src.models.llm import get_llm, create_arya_agent

__all__ = ["load_stt_model", "load_tts_model", "get_llm", "create_arya_agent"]
