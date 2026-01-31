"""
TTS (Text-to-Speech) using Sarvam AI for Sarah Voice Agent.
"""

import io
import asyncio
import requests
import numpy as np
import av
import os
import base64
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError

logger = setup_logging("Models-TTS-Sarvam")

class SarvamTTS:
    """
    Sarah's voice using Sarvam AI's Bulbul model.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.sarvam.ai/text-to-speech"
        self.voice = "arya" # Using latest high-quality speaker
        logger.info(f"🔊 Sarvam TTS initialized with voice: {self.voice}")

    async def stream_tts(self, text: str):
        """
        Stream audio from Sarvam AI.
        """
        if not text: return

        try:
            # Updated payload for Sarvam Bulbul v2 API
            payload = {
                "inputs": [text],
                "target_language_code": "en-IN",
                "speaker": self.voice,
                "model": "bulbul:v2",
                "speech_sample_rate": 16000,
                "enable_preprocessing": True
            }
            
            headers = {
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(self.url, json=payload, headers=headers, timeout=10)
            )
            
            if response.status_code == 200:
                # Sarvam returns an 'audios' list of base64 strings
                audios = response.json().get("audios", [])
                if not audios:
                    logger.warning("Sarvam returned no audio in response")
                    return
                
                audio_bytes = base64.b64decode(audios[0])
                
                # Decode to PCM using PyAV
                mp3_data = io.BytesIO(audio_bytes)
                container = av.open(mp3_data)
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
                
                for frame in container.decode(stream):
                    resampled_frames = resampler.resample(frame)
                    for f in resampled_frames:
                        array = f.to_ndarray().reshape(-1)
                        # Speed up playback onset
                        yield (16000, array.astype(np.int16))
                container.close()
            else:
                logger.error(f"Sarvam TTS Error ({response.status_code}): {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Sarvam TTS Exception: {e}", exc_info=True)

def load_tts_model():
    from src.config import get_config
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.warning("SARVAM_API_KEY not found, falling back to Edge-TTS")
        from src.models.tts import EdgeTTSWrapper
        return EdgeTTSWrapper()
    
    return SarvamTTS(api_key)
