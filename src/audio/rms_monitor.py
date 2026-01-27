"""
RMS (Root Mean Square) Monitor for Voice Activity Detection.

This module provides continuous audio level monitoring for:
- Voice activity detection (VAD)
- Barge-in detection during TTS playback
- Speech energy analysis for adaptive thresholds
"""

import audioop
from collections import deque
from dataclasses import dataclass, field
from typing import List, Callable, Optional
from datetime import datetime

from src.utils.logger import setup_logging

logger = setup_logging("Audio-RMS")


@dataclass
class RMSReading:
    """Single RMS measurement with metadata."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    is_speech: bool = False


@dataclass
class RMSAnalysis:
    """Analysis result from RMS processing."""
    current_rms: float
    average_rms: float
    peak_rms: float
    is_speaking: bool
    speech_started: bool
    speech_ended: bool
    energy_level: str  # "silence", "low", "normal", "loud"


class RMSMonitor:
    """
    Continuous RMS monitoring for voice activity detection and barge-in.
    
    Features:
    - Rolling window for smoothed RMS analysis
    - Speech onset/offset detection
    - Energy level classification
    - Callback system for event handling
    """
    
    def __init__(
        self, 
        silence_threshold: float = 1000,
        barge_in_threshold: float = 1500,
        window_size: int = 25,  # ~500ms at 20ms chunks
        sample_width: int = 2
    ):
        """
        Initialize RMS monitor.
        
        Args:
            silence_threshold: RMS below this is considered silence
            barge_in_threshold: RMS above this during playback triggers barge-in
            window_size: Number of readings for rolling average
            sample_width: Audio sample width in bytes (2 for int16)
        """
        self.silence_threshold = silence_threshold
        self.barge_in_threshold = barge_in_threshold
        self.sample_width = sample_width
        
        # Rolling window
        self._rms_history: deque = deque(maxlen=window_size)
        self._peak_rms: float = 0.0
        
        # State tracking
        self._is_user_speaking: bool = False
        self._consecutive_speech_frames: int = 0
        self._consecutive_silence_frames: int = 0
        
        # Frame thresholds for speech detection (hysteresis)
        self._speech_onset_frames: int = 2   # ~40ms to confirm speech start
        self._speech_offset_frames: int = 10 # ~200ms to confirm speech end
        
        # Callbacks
        self._on_speech_start: List[Callable] = []
        self._on_speech_end: List[Callable] = []
        self._on_barge_in: List[Callable] = []
        
        logger.debug(
            f"RMS Monitor initialized: "
            f"silence_threshold={silence_threshold}, "
            f"barge_in_threshold={barge_in_threshold}"
        )
    
    def process_audio(self, pcm_audio: bytes) -> RMSAnalysis:
        """
        Process audio chunk and return RMS analysis.
        
        Args:
            pcm_audio: Raw PCM audio bytes (int16)
            
        Returns:
            RMSAnalysis with current audio state
        """
        # Calculate RMS
        try:
            rms = audioop.rms(pcm_audio, self.sample_width)
        except audioop.error:
            rms = 0.0
        
        # Update history
        self._rms_history.append(rms)
        self._peak_rms = max(self._peak_rms, rms)
        
        # Calculate averages
        avg_rms = sum(self._rms_history) / len(self._rms_history) if self._rms_history else 0.0
        
        # Detect speech state changes with hysteresis
        was_speaking = self._is_user_speaking
        
        if rms > self.silence_threshold:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0
            
            if self._consecutive_speech_frames >= self._speech_onset_frames:
                self._is_user_speaking = True
        else:
            self._consecutive_silence_frames += 1
            self._consecutive_speech_frames = 0
            
            if self._consecutive_silence_frames >= self._speech_offset_frames:
                self._is_user_speaking = False
        
        # Determine state transitions
        speech_started = self._is_user_speaking and not was_speaking
        speech_ended = not self._is_user_speaking and was_speaking
        
        # Trigger callbacks
        if speech_started:
            for callback in self._on_speech_start:
                callback()
        if speech_ended:
            for callback in self._on_speech_end:
                callback()
        
        # Classify energy level
        energy_level = self._classify_energy(rms)
        
        return RMSAnalysis(
            current_rms=rms,
            average_rms=avg_rms,
            peak_rms=self._peak_rms,
            is_speaking=self._is_user_speaking,
            speech_started=speech_started,
            speech_ended=speech_ended,
            energy_level=energy_level
        )
    
    def check_barge_in(self, pcm_audio: bytes) -> bool:
        """
        Check if audio represents a barge-in attempt.
        
        Use this method while TTS is playing to detect user interruption.
        
        Args:
            pcm_audio: Raw PCM audio bytes
            
        Returns:
            True if barge-in detected
        """
        try:
            rms = audioop.rms(pcm_audio, self.sample_width)
        except audioop.error:
            return False
        
        is_barge_in = rms > self.barge_in_threshold
        
        if is_barge_in:
            logger.info(f"🚨 Barge-in detected! RMS: {rms}")
            for callback in self._on_barge_in:
                callback()
        
        return is_barge_in
    
    def _classify_energy(self, rms: float) -> str:
        """Classify current energy level."""
        if rms < self.silence_threshold * 0.5:
            return "silence"
        elif rms < self.silence_threshold:
            return "low"
        elif rms < self.barge_in_threshold:
            return "normal"
        else:
            return "loud"
    
    def register_callback(
        self, 
        event: str, 
        callback: Callable
    ) -> None:
        """
        Register callback for audio events.
        
        Args:
            event: One of "speech_start", "speech_end", "barge_in"
            callback: Function to call when event occurs
        """
        if event == "speech_start":
            self._on_speech_start.append(callback)
        elif event == "speech_end":
            self._on_speech_end.append(callback)
        elif event == "barge_in":
            self._on_barge_in.append(callback)
        else:
            logger.warning(f"Unknown event type: {event}")
    
    def reset(self) -> None:
        """Reset monitor state for new session."""
        self._rms_history.clear()
        self._peak_rms = 0.0
        self._is_user_speaking = False
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        logger.debug("RMS Monitor reset")
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "current_rms": self._rms_history[-1] if self._rms_history else 0,
            "average_rms": sum(self._rms_history) / len(self._rms_history) if self._rms_history else 0,
            "peak_rms": self._peak_rms,
            "is_speaking": self._is_user_speaking,
            "history_length": len(self._rms_history)
        }
    
    def adjust_threshold(self, new_silence_threshold: float) -> None:
        """Dynamically adjust silence threshold."""
        self.silence_threshold = new_silence_threshold
        logger.debug(f"Silence threshold adjusted to: {new_silence_threshold}")
