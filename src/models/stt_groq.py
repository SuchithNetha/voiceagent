"""
Groq-powered STT (Speech-to-Text) for Sarah Voice Agent.

Uses Groq's incredibly fast Whisper-large-v3 API for near-real-time
performance without heavy local models.
"""

import os
import io
import wave
import asyncio
import numpy as np
from typing import Tuple, Optional
from groq import Groq
from src.config import get_config
from src.utils.logger import setup_logging
from src.utils.exceptions import TranscriptionError
from src.utils.retry import async_retry

logger = setup_logging("Models-STT-Groq")
app_config = get_config()

class GroqSTT:
    """Fast STT implementation using Groq's Whisper API."""
    
    def __init__(self):
        self.client = Groq(api_key=app_config.GROQ_API_KEY)
        self.model = "whisper-large-v3"
        logger.info(f"🎤 Groq STT initialized with {self.model}")
        
    @async_retry(tries=3, delay=0.2, backoff=2.0)
    async def stt(self, audio: Tuple[int, np.ndarray]) -> str:
        """
        Transcribe audio using Groq API.
        
        Args:
            audio: Tuple of (sample_rate, audio_numpy_array)
            
        Returns:
            Transcription text
        """
        sample_rate, audio_data = audio
        
        if len(audio_data) == 0:
            return ""
            
        try:
            # 1. Convert numpy array to WAV bytes in memory
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            buffer.seek(0)
            
            # 2. Call Groq API
            response = await asyncio.to_thread(
                self.client.audio.transcriptions.create,
                file=("audio.wav", buffer),
                model=self.model,
                response_format="text",
                language="en"
            )
            
            transcription = response.strip()
            if transcription:
                logger.debug(f"📝 Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"❌ Groq STT error: {e}")
            raise TranscriptionError(e)

def load_groq_stt_model():
    """Load the Groq STT model wrapper."""
    try:
        if not app_config.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found. Cannot use Groq STT.")
            return None
        return GroqSTT()
    except Exception as e:
        logger.error(f"Failed to load Groq STT: {e}")
        return None
