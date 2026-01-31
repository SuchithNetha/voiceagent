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
        # Using a modern, natural AvaNeural voice (US English - perfect for an international real estate agent)
        self.voice = "en-US-AvaNeural" 
        logger.info(f"🔊 EdgeTTS initialized with voice: {self.voice}")

    async def stream_tts(self, text: str):
        """
        Convert text to speech and yield audio chunks with minimal latency.
        Decodes audio chunks as they arrive from the Edge-TTS stream.
        """
        if not text:
            return

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            
            # Initialize PyAV decoder for incremental decoding
            codec = av.CodecContext.create('mp3', 'r')
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=16000,
            )
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Feed chunk to codec and get frames
                    packets = codec.parse(chunk["data"])
                    for packet in packets:
                        frames = codec.decode(packet)
                        for frame in frames:
                            resampled_frames = resampler.resample(frame)
                            for f in resampled_frames:
                                array = f.to_ndarray().reshape(-1)
                                yield (16000, array.astype(np.int16))
            
            # Flush the codec
            packets = codec.parse(b'')
            for packet in packets:
                frames = codec.decode(packet)
                for frame in frames:
                    resampled_frames = resampler.resample(frame)
                    for f in resampled_frames:
                        array = f.to_ndarray().reshape(-1)
                        yield (16000, array.astype(np.int16))

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
