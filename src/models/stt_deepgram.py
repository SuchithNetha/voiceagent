"""
Deepgram STT with Speaker Diarization for Arya Voice Agent.

This solves the MULTI-SPEAKER PROBLEM:
- Background voices are separated from the primary speaker
- Only the caller's words are transcribed
- Background noise like "DINNER TIME!" gets filtered out

HOW IT WORKS:
1. Audio comes in with mixed voices
2. Deepgram's ML model identifies different speakers
3. Each word is labeled: Speaker 0, Speaker 1, etc.
4. Speaker 0 = closest to microphone = the caller
5. We only return Speaker 0's words!

Example:
  Raw audio: "I want apartment DINNER TIME in Salamanca"
  After diarization:
    - Speaker 0: "I want apartment in Salamanca"  ← We use this!
    - Speaker 1: "DINNER TIME"                    ← Filtered out!
"""

import os
import io
import wave
import asyncio
import numpy as np
from typing import Tuple, Optional, List
from src.utils.logger import setup_logging
from src.utils.exceptions import TranscriptionError
from src.utils.retry import async_retry

logger = setup_logging("Models-STT-Deepgram")


class DeepgramSTT:
    """
    STT with Speaker Diarization using Deepgram Nova-2.
    
    Key Feature: Separates multiple speakers and returns only the primary speaker's words.
    """
    
    def __init__(self):
        try:
            from deepgram import DeepgramClient
            
            self.api_key = os.getenv("DEEPGRAM_API_KEY")
            if not self.api_key:
                raise ValueError("DEEPGRAM_API_KEY not set")
            
            self.client = DeepgramClient(self.api_key)
            self.model = "nova-2"  # Deepgram's best model
            logger.info(f"🎤 Deepgram STT initialized with {self.model} + Speaker Diarization")
            
        except ImportError:
            logger.error("deepgram-sdk not installed. Run: pip install deepgram-sdk")
            raise
    
    @async_retry(tries=3, delay=0.2, backoff=2.0)
    async def stt(self, audio: Tuple[int, np.ndarray]) -> str:
        """
        Transcribe audio with speaker diarization.
        
        Args:
            audio: Tuple of (sample_rate, audio_numpy_array)
            
        Returns:
            Transcription text (PRIMARY SPEAKER ONLY!)
        """
        from deepgram import PrerecordedOptions
        
        sample_rate, audio_data = audio
        
        if len(audio_data) == 0:
            return ""
        
        try:
            # 1. Convert numpy array to WAV bytes
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            # 2. Configure Deepgram with DIARIZATION
            options = PrerecordedOptions(
                model=self.model,
                language="en",
                smart_format=True,      # Better punctuation
                diarize=True,           # 🔑 THE KEY FEATURE! Separates speakers
                punctuate=True,
                utterances=True,        # Group by speaker turns
            )
            
            # 3. Send to Deepgram
            logger.debug("📤 Sending audio to Deepgram with diarization...")
            
            response = await asyncio.to_thread(
                self._transcribe_sync,
                audio_bytes,
                options
            )
            
            # 4. Extract PRIMARY SPEAKER only (Speaker 0)
            transcription = self._extract_primary_speaker(response)
            
            if transcription:
                logger.debug(f"📝 Primary speaker transcription: {transcription}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"❌ Deepgram STT error: {e}")
            raise TranscriptionError(e)
    
    def _transcribe_sync(self, audio_bytes: bytes, options):
        """Synchronous transcription call (run in thread)."""
        from deepgram import FileSource
        
        payload: FileSource = {"buffer": audio_bytes}
        response = self.client.listen.rest.v("1").transcribe_file(payload, options)
        return response
    
    def _extract_primary_speaker(self, response) -> str:
        """
        Extract ONLY the primary speaker's words.
        
        Speaker 0 = Closest to microphone = The actual caller
        Speaker 1, 2, etc. = Background voices = IGNORED
        
        Also detects GARBLED speech (overlapping speakers, low confidence)
        and returns "[UNCLEAR]" so the LLM can ask user to repeat.
        
        This is how we solve the multi-speaker problem!
        """
        try:
            # Get the words with speaker labels
            alternatives = response.results.channels[0].alternatives[0]
            
            # Check overall confidence
            confidence = getattr(alternatives, 'confidence', 0.9)
            
            # Check if we have word-level data with speakers
            if hasattr(alternatives, 'words') and alternatives.words:
                words = alternatives.words
                
                # Count speakers for logging
                speakers = set(w.speaker for w in words if hasattr(w, 'speaker'))
                num_speakers = len(speakers)
                
                if num_speakers > 1:
                    logger.info(f"🎭 Detected {num_speakers} speakers! Filtering to primary...")
                
                # Filter to Speaker 0 only (the phone holder)
                primary_words = []
                other_words = []
                
                for word in words:
                    speaker = getattr(word, 'speaker', 0)
                    if speaker == 0:  # Primary speaker only!
                        primary_words.append(word.word)
                    else:
                        other_words.append(word.word)
                        logger.debug(f"🔇 Filtered out (Speaker {speaker}): '{word.word}'")
                
                result = " ".join(primary_words)
                full_transcript = " ".join(w.word for w in words)
                
                # Log what we filtered
                if result != full_transcript:
                    logger.info(f"🎯 Original: '{full_transcript}'")
                    logger.info(f"🎯 Filtered: '{result}'")
                
                # 🔑 GARBLED DETECTION - Ask to repeat if:
                # 1. Too many speakers (3+) with similar word counts = chaos
                # 2. Primary speaker has very few words but others have many
                # 3. Low confidence overall
                
                if num_speakers >= 3:
                    logger.warning("⚠️ 3+ speakers detected - likely garbled")
                    return "[UNCLEAR]"
                
                if len(primary_words) < 2 and len(other_words) > 3:
                    logger.warning("⚠️ Primary speaker drowned out by background")
                    return "[UNCLEAR]"
                
                if confidence < 0.5:
                    logger.warning(f"⚠️ Low confidence ({confidence:.2f}) - likely garbled")
                    return "[UNCLEAR]"
                
                # If result is empty but there were words, something went wrong
                if not result.strip() and len(words) > 0:
                    logger.warning("⚠️ No primary speaker words extracted")
                    return "[UNCLEAR]"
                
                return result.strip()
            
            # Fallback: no word-level data, use full transcript
            transcript = alternatives.transcript.strip()
            
            # Check if transcript seems garbled (too short or low confidence)
            if confidence < 0.5 or len(transcript.split()) < 1:
                return "[UNCLEAR]"
                
            return transcript
            
        except Exception as e:
            logger.warning(f"Speaker extraction failed, using full transcript: {e}")
            try:
                return response.results.channels[0].alternatives[0].transcript.strip()
            except:
                return "[UNCLEAR]"
    
    async def stt_with_confidence(self, audio: Tuple[int, np.ndarray]) -> Tuple[str, float]:
        """
        Transcribe with confidence score.
        
        Returns:
            Tuple of (transcription, confidence 0.0-1.0)
        """
        from deepgram import PrerecordedOptions
        
        sample_rate, audio_data = audio
        
        if len(audio_data) == 0:
            return "", 0.0
        
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            options = PrerecordedOptions(
                model=self.model,
                language="en",
                smart_format=True,
                diarize=True,
                punctuate=True,
            )
            
            response = await asyncio.to_thread(
                self._transcribe_sync,
                audio_bytes,
                options
            )
            
            # Get transcription and confidence
            alternatives = response.results.channels[0].alternatives[0]
            transcription = self._extract_primary_speaker(response)
            confidence = getattr(alternatives, 'confidence', 0.9)
            
            return transcription, confidence
            
        except Exception as e:
            logger.error(f"❌ Deepgram STT error: {e}")
            return "", 0.0


def load_deepgram_stt_model():
    """Load the Deepgram STT model with speaker diarization."""
    try:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            logger.warning("DEEPGRAM_API_KEY not found. Cannot use Deepgram STT.")
            return None
        return DeepgramSTT()
    except Exception as e:
        logger.error(f"Failed to load Deepgram STT: {e}")
        return None
