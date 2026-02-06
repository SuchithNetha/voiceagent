"""
Streaming Deepgram STT for Arya Voice Agent.
Optimized for real-time telephony with ultra-low latency.
"""

import os
import asyncio
import json
import numpy as np
from typing import Callable, Optional
from src.utils.logger import setup_logging
try:
    from deepgram import (
        DeepgramClient,
        LiveOptions,
        LiveTranscriptionEvents,
        DeepgramClientOptions
    )
except ImportError:
    # Fallback for SDK version variations where these aren't exported at top level
    from deepgram import DeepgramClient, DeepgramClientOptions
    from deepgram.clients.listen.v1.websocket.options import LiveOptions
    from deepgram.clients.listen.v1.websocket.events import LiveTranscriptionEvents
    logger.warning("⚠️ Deepgram imports required explicit sub-client paths")

logger = setup_logging("Models-STT-DeepgramStream")

class DeepgramStreamer:
    """
    Handles a persistent WebSocket connection to Deepgram for streaming STT.
    """
    def __init__(self, api_key: str, on_transcript: Callable[[str, bool], None]):
        """
        Args:
            api_key: Deepgram API Key
            on_transcript: Callback(transcript_text, is_final)
        """
        self.api_key = api_key
        config = DeepgramClientOptions(
            options={"keepalive": "true"}
        )
        self.client = DeepgramClient(api_key, config)
        self.on_transcript = on_transcript
        self.dg_connection = None
        self.is_active = False
        
    async def start(self):
        """Start the streaming connection."""
        if self.dg_connection:
            return
            
        self.dg_connection = self.client.listen.live.v("1")
        
        # Define event handlers
        def on_message(self_dg, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not sentence:
                return
                
            is_final = result.is_final
            speech_final = result.speech_final
            start_time = result.start # Stream-relative start time in seconds
            
            # Extract speaker ID if diarization is enabled
            speaker_id = 0
            if hasattr(result.channel.alternatives[0], 'words') and result.channel.alternatives[0].words:
                speaker_id = result.channel.alternatives[0].words[0].speaker
            elif hasattr(result, 'speaker'):
                speaker_id = result.speaker
            
            # Use on_transcript callback with timing info and speaker ID
            # Callback(text, is_final, speech_final, start_time, speaker_id)
            if speech_final or is_final:
                logger.debug(f"📥 Deepgram Final [{start_time:.2f}s, Spk:{speaker_id}]: {sentence}")
                self.on_transcript(sentence, True, speech_final, start_time, speaker_id)
            else:
                logger.log(5, f"📥 Deepgram Partial: {sentence}")
                self.on_transcript(sentence, False, False, start_time, speaker_id)

        def on_error(self_dg, error, **kwargs):
            logger.error(f"❌ Deepgram Stream Error: {error}")

        def on_close(self_dg, close, **kwargs):
            logger.info("🔌 Deepgram Stream Closed")
            self.is_active = False

        # Register handlers
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        self.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        
        options = LiveOptions(
            model="nova-2",
            language="en",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=8000, # Twilio uses 8kHz
            interim_results=True,
            utterance_end_ms="1500", # Wait longer before closing utterance
            vad_events=True,
            endpointing=300, # Faster results for short utterances
            diarize=True,    # Still use diarization to filter background
        )
        
        try:
            # Sync call in thread to avoid blocking loop
            if await asyncio.to_thread(self.dg_connection.start, options):
                self.is_active = True
                logger.info("✅ Deepgram Streaming STT connected")
            else:
                logger.error("❌ Failed to start Deepgram Stream")
        except Exception as e:
            logger.error(f"❌ Deepgram Stream Init Exception: {e}")
            self.is_active = False

    async def send_audio(self, pcm_data: bytes):
        """Sends audio chunks to Deepgram."""
        if self.is_active and self.dg_connection:
            try:
                # Use to_thread to avoid blocking the main hearing loop
                await asyncio.to_thread(self.dg_connection.send, pcm_data)
            except Exception as e:
                logger.error(f"Error sending audio to Deepgram: {e}")
                self.is_active = False

    async def stop(self):
        """Close the connection."""
        if self.dg_connection:
            try:
                await asyncio.to_thread(self.dg_connection.finish)
            except:
                pass
            self.dg_connection = None
            self.is_active = False
            logger.info("🛑 Deepgram Stream cleaned up")

def load_deepgram_streamer(on_transcript: Callable[[str, bool, bool, float], None]):
    """Helper to initialize streamer with API key."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("DEEPGRAM_API_KEY missing")
        return None
    return DeepgramStreamer(api_key, on_transcript)
