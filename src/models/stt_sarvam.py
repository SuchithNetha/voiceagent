"""
STT (Speech-to-Text) using Sarvam AI for Sarah Voice Agent.
Ultra-fast transcription optimized for phone conversations.
"""

import os
import io
import asyncio
import requests
import numpy as np
import wave
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
        self.model = "saarika:v2.5"  # Latest Sarvam STT model
        self.language_code = "en-IN"  # English (India) - works for most English
        logger.info(f"🎤 Sarvam STT initialized with model: {self.model}")

    async def stt(self, audio: tuple) -> str:
        """
        Transcribe audio using Sarvam AI.
        """
        sample_rate, audio_data = audio
        
        # Validate input
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data received")
            return ""
        
        try:
            # Sarvam works best with 16kHz audio
            # If we received different sample rate, log it
            if sample_rate != 16000:
                logger.debug(f"Audio sample rate: {sample_rate}Hz (Sarvam prefers 16kHz)")
            
            # Convert numpy array to WAV bytes using standard wave library
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            buffer.seek(0)
            audio_size = buffer.getbuffer().nbytes
            logger.debug(f"Audio buffer size: {audio_size} bytes, duration: {len(audio_data)/sample_rate:.2f}s")
            
            # Call Sarvam API with all required parameters
            files = {"file": ("audio.wav", buffer, "audio/wav")}
            headers = {"api-subscription-key": self.api_key}
            data = {
                "model": self.model,
                "language_code": self.language_code
            }
            
            # Use run_in_executor for sync requests call
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: requests.post(
                    self.url, 
                    files=files, 
                    data=data,
                    headers=headers, 
                    timeout=15
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("transcript", "")
                if text:
                    logger.debug(f"📝 Sarvam transcribed: {text[:50]}...")
                else:
                    logger.warning("Sarvam returned empty transcript")
                return text
            else:
                logger.error(f"Sarvam STT Error ({response.status_code}): {response.text}")
                # Log full response for debugging
                try:
                    error_json = response.json()
                    logger.error(f"Error details: {error_json}")
                except:
                    pass
                return ""
                
        except requests.exceptions.Timeout:
            logger.error("❌ Sarvam STT Timeout (>15s)")
            return ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Sarvam STT Connection Error: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Sarvam STT Exception: {e}", exc_info=True)
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

