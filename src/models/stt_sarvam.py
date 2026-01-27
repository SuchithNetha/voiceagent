"""
STT (Speech-to-Text) using Sarvam AI for Sarah Voice Agent.
Ultra-fast transcription optimized for phone conversations.
"""

import io
import asyncio
import requests
import numpy as np
import scipy.io.wavfile as wavfile
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError

logger = setup_logging("Models-STT-Sarvam")

class SarvamSTT:
    """
    Sarah's ears using Sarvam AI's Speech-to-Text API.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.sarvam.ai/speech-to-text"
        logger.info("🎤 Sarvam STT initialized")

    async def stt(self, audio: tuple) -> str:
        """
        Transcribe audio using Sarvam AI.
        """
        sample_rate, audio_data = audio
        
        try:
            # Convert numpy array to WAV bytes
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio_data)
            buffer.seek(0)
            
            # Call Sarvam API
            files = {"file": ("audio.wav", buffer, "audio/wav")}
            headers = {"api-subscription-key": self.api_key}
            
            # Use run_in_executor for sync requests call
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: requests.post(self.url, files=files, headers=headers)
            )
            
            if response.status_code == 200:
                text = response.json().get("transcript", "")
                return text
            else:
                logger.error(f"Sarvam STT Error: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Sarvam STT Exception: {e}")
            return ""

def load_stt_model():
    """Initialize Sarvam STT."""
    from src.config import get_config
    config = get_config()
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.warning("SARVAM_API_KEY not found, falling back to Groq STT")
        from src.models.stt_groq import load_groq_stt_model
        return load_groq_stt_model()
        
    return SarvamSTT(api_key)
