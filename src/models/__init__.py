import os
from src.utils.logger import setup_logging

logger = setup_logging("Models-Init")

def load_stt_model():
    """
    Load STT model based on config.
    
    Options (set STT_MODEL in .env):
    - "groq"    : Fast Whisper API (default, no speaker separation)
    - "deepgram": Nova-2 with SPEAKER DIARIZATION (filters background speakers!)
    - "sarvam"  : Indian language support
    """
    stt_type = os.getenv("STT_MODEL", "groq").lower()
    
    if stt_type == "deepgram":
        # 🎯 SPEAKER DIARIZATION - Solves multi-speaker problem!
        logger.info("🎭 Loading Deepgram STT with Speaker Diarization...")
        from src.models.stt_deepgram import load_deepgram_stt_model
        model = load_deepgram_stt_model()
        if model:
            return model
        # Fallback to Groq if Deepgram fails
        logger.warning("⚠️ Deepgram failed, falling back to Groq STT")
        from src.models.stt_groq import load_groq_stt_model
        return load_groq_stt_model()
    
    elif stt_type == "sarvam":
        from src.models.stt_sarvam import load_stt_model as load
        return load()
    
    else:  # Default: Groq
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
