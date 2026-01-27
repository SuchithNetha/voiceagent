"""
Lightweight TTS (Text-to-Speech) using Edge-TTS for Sarah Voice Agent.
Provides high-quality speech without the heavy 100MB+ local model overhead.
"""

import io
import asyncio
import edge_tts
import numpy as np
import av
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError

logger = setup_logging("Models-TTS-Lite")

class EdgeTTSWrapper:
    """
    Sarah's voice using Edge-TTS API (Free, high-quality, 0MB local RAM).
    """
    def __init__(self):
        # Using a warm, natural SoniaNeural voice (British English - professional real estate tone)
        self.voice = "en-GB-SoniaNeural" 
        logger.info(f"🔊 EdgeTTS initialized with voice: {self.voice}")

    async def stream_tts(self, text: str):
        """
        Convert text to speech and yield audio chunks.
        Matches the interface expected by SarahAgent.
        """
        if not text:
            return

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            
            # Use a buffer to collect the MP3 stream from Edge-TTS
            mp3_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data.write(chunk["data"])
            
            mp3_data.seek(0)
            
            # Decode MP3 to 16kHz PCM using PyAV
            try:
                container = av.open(mp3_data)
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(
                    format='s16',
                    layout='mono',
                    rate=16000,
                )
                
                for frame in container.decode(stream):
                    resampled_frames = resampler.resample(frame)
                    for f in resampled_frames:
                        # Convert to numpy and yield
                        array = f.to_ndarray().reshape(-1)
                        # FastRTC/Twilio Expects int16
                        yield (16000, array.astype(np.int16))
                
                container.close()
            except Exception as e:
                logger.error(f"Decoding error: {e}")
            
        except Exception as e:
            logger.error(f"❌ EdgeTTS Streaming error: {e}")

def load_tts_model():
    """
    Initialize and return the lightweight Edge-TTS wrapper.
    """
    logger.info("🔊 Loading Lightweight Edge-TTS (API-based, 0MB RAM)...")
    try:
        return EdgeTTSWrapper()
    except Exception as e:
        logger.critical(f"❌ Failed to load TTS model: {e}")
        raise ModelLoadError("EdgeTTS", original_error=e)
