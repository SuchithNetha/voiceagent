"""
Adaptive Voice Activity Detection (VAD) for Sarah Voice Agent.

This module implements adaptive silence detection that adjusts thresholds based on:
- Recent speech energy levels
- Speaker cadence patterns
- Conversation context
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from src.utils.logger import setup_logging

logger = setup_logging("Audio-AdaptiveVAD")


@dataclass
class SpeechSegment:
    """Represents a detected speech segment."""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: int = 0
    avg_rms: float = 0.0
    word_density: float = 1.0  # Words per second estimate


class AdaptiveSilenceDetector:
    """
    Adaptive silence detection with dynamic threshold adjustment.
    
    Reduces the fixed 800ms silence limit to 600ms baseline, with the ability
    to adapt based on:
    - Speaker's natural pause patterns
    - Speech energy levels
    - Question/statement context
    
    This makes the agent feel more attentive and responsive.
    """
    
    def __init__(
        self,
        base_silence_ms: int = 600,      # Down from 800ms
        min_silence_ms: int = 400,        # Minimum for fast talkers
        max_silence_ms: int = 900,        # Maximum for deliberate speech
        adaptation_window: int = 20       # Number of pauses to analyze
    ):
        """
        Initialize adaptive detector.
        
        Args:
            base_silence_ms: Default silence threshold
            min_silence_ms: Minimum allowed threshold
            max_silence_ms: Maximum allowed threshold
            adaptation_window: Number of recent pauses to analyze for adaptation
        """
        self.base_silence_ms = base_silence_ms
        self.min_silence_ms = min_silence_ms
        self.max_silence_ms = max_silence_ms
        
        # Adaptation tracking
        self._recent_rms_values: deque = deque(maxlen=50)
        self._recent_pause_lengths: deque = deque(maxlen=adaptation_window)
        self._speech_segments: deque = deque(maxlen=adaptation_window)
        
        # Current adapted values
        self._current_threshold_ms: int = base_silence_ms
        self._speaker_pace: str = "normal"  # "fast", "normal", "slow"
        
        # State tracking
        self._current_segment: Optional[SpeechSegment] = None
        self._last_speech_end: Optional[datetime] = None
        self._accumulated_silence_ms: int = 0
        
        logger.info(
            f"AdaptiveVAD initialized: base={base_silence_ms}ms, "
            f"range=[{min_silence_ms}, {max_silence_ms}]ms"
        )
    
    def update(self, rms: float, is_speech: bool, packet_duration_ms: int = 20) -> dict:
        """
        Update detector state with new audio frame.
        
        Args:
            rms: Current RMS value
            is_speech: Whether current frame is speech
            packet_duration_ms: Duration of audio packet
            
        Returns:
            Dict with threshold info and speech state
        """
        self._recent_rms_values.append(rms)
        now = datetime.now()
        
        if is_speech:
            # Track speech segment
            if self._current_segment is None:
                self._current_segment = SpeechSegment(start_time=now)
                
                # Record pause length if we had silence before
                if self._last_speech_end:
                    pause_ms = (now - self._last_speech_end).total_seconds() * 1000
                    if pause_ms < 5000:  # Only track reasonable pauses
                        self._recent_pause_lengths.append(pause_ms)
            
            self._accumulated_silence_ms = 0
        else:
            # Track silence
            self._accumulated_silence_ms += packet_duration_ms
            
            # End current speech segment
            if self._current_segment is not None:
                self._current_segment.end_time = now
                self._current_segment.duration_ms = int(
                    (now - self._current_segment.start_time).total_seconds() * 1000
                )
                self._current_segment.avg_rms = (
                    sum(self._recent_rms_values) / len(self._recent_rms_values)
                    if self._recent_rms_values else 0
                )
                self._speech_segments.append(self._current_segment)
                self._current_segment = None
                self._last_speech_end = now
        
        # Get adaptive threshold
        dynamic_threshold = self.get_dynamic_threshold()
        
        # Check if pause is complete
        pause_complete = (
            self._accumulated_silence_ms >= dynamic_threshold
            and self._last_speech_end is not None
        )
        
        return {
            "current_threshold_ms": dynamic_threshold,
            "accumulated_silence_ms": self._accumulated_silence_ms,
            "pause_complete": pause_complete,
            "speaker_pace": self._speaker_pace,
            "adaptation_active": len(self._speech_segments) > 3
        }
    
    def get_dynamic_threshold(
        self, 
        is_question_context: bool = False,
        force_responsive: bool = False
    ) -> int:
        """
        Calculate dynamic silence threshold based on context.
        
        Args:
            is_question_context: If True, agent just asked a question (allow longer pause)
            force_responsive: If True, use minimum threshold for quick response
            
        Returns:
            Silence threshold in milliseconds
        """
        if force_responsive:
            return self.min_silence_ms
        
        base = self.base_silence_ms
        
        # Adjust based on detected speaker pace
        if len(self._speech_segments) >= 3:
            avg_segment_duration = sum(
                seg.duration_ms for seg in self._speech_segments
            ) / len(self._speech_segments)
            
            if avg_segment_duration < 500:
                # Short utterances = fast speaker
                self._speaker_pace = "fast"
                base = int(base * 0.8)  # 20% reduction
            elif avg_segment_duration > 2000:
                # Long utterances = deliberate speaker
                self._speaker_pace = "slow"
                base = int(base * 1.2)  # 20% increase
            else:
                self._speaker_pace = "normal"
        
        # Adjust based on recent pause patterns
        if len(self._recent_pause_lengths) >= 3:
            avg_pause = sum(self._recent_pause_lengths) / len(self._recent_pause_lengths)
            # If user naturally takes short pauses, reduce threshold
            if avg_pause < 400:
                base = int(base * 0.85)
            elif avg_pause > 800:
                base = int(base * 1.1)
        
        # Context adjustments
        if is_question_context:
            # Give user more time to formulate answer
            base = int(base * 1.3)
        
        # Energy-based adjustment
        if self._recent_rms_values:
            recent_avg = sum(list(self._recent_rms_values)[-5:]) / min(5, len(self._recent_rms_values))
            overall_avg = sum(self._recent_rms_values) / len(self._recent_rms_values)
            
            # If recent energy is dropping, user might be trailing off
            if overall_avg > 0 and recent_avg < overall_avg * 0.5:
                base = int(base * 0.9)  # Slight reduction
        
        # Clamp to bounds
        self._current_threshold_ms = max(
            self.min_silence_ms, 
            min(self.max_silence_ms, base)
        )
        
        return self._current_threshold_ms
    
    def is_phrase_complete(self, min_audio_ms: int = 200) -> bool:
        """
        Check if current phrase is complete (ready for processing).
        
        Args:
            min_audio_ms: Minimum audio duration to consider valid
            
        Returns:
            True if phrase is complete
        """
        if self._last_speech_end is None:
            return False
        
        # Check if we have enough accumulated silence
        if self._accumulated_silence_ms < self._current_threshold_ms:
            return False
        
        # Check if last segment was long enough
        if self._speech_segments:
            last_segment = self._speech_segments[-1]
            if last_segment.duration_ms < min_audio_ms:
                return False
        
        return True
    
    def reset(self) -> None:
        """Reset detector for new conversation."""
        self._recent_rms_values.clear()
        self._recent_pause_lengths.clear()
        self._speech_segments.clear()
        self._current_segment = None
        self._last_speech_end = None
        self._accumulated_silence_ms = 0
        self._current_threshold_ms = self.base_silence_ms
        self._speaker_pace = "normal"
        logger.debug("AdaptiveVAD reset")
    
    def get_debug_info(self) -> dict:
        """Get debug information about current state."""
        return {
            "current_threshold_ms": self._current_threshold_ms,
            "base_silence_ms": self.base_silence_ms,
            "speaker_pace": self._speaker_pace,
            "accumulated_silence_ms": self._accumulated_silence_ms,
            "num_segments_analyzed": len(self._speech_segments),
            "avg_pause_length": (
                sum(self._recent_pause_lengths) / len(self._recent_pause_lengths)
                if self._recent_pause_lengths else 0
            ),
            "avg_rms": (
                sum(self._recent_rms_values) / len(self._recent_rms_values)
                if self._recent_rms_values else 0
            )
        }
