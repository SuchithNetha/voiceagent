"""
Enhanced Voice Activity Detection using WebRTC VAD.

WebRTC VAD is a machine learning model trained to distinguish
human speech from noise (coughs, sneezes, background sounds).

This provides much better accuracy than simple RMS thresholds.
"""

import struct
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum

from src.utils.logger import setup_logging

logger = setup_logging("Audio-WebRTCVAD")

# Try to import webrtcvad, fall back to RMS-only if not available
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
    logger.info("✅ WebRTC VAD available - using ML-based speech detection")
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.warning("⚠️ webrtcvad not installed - falling back to RMS-only detection")


class VADAggressiveness(IntEnum):
    """
    WebRTC VAD aggressiveness levels.
    
    Higher = more aggressive filtering of non-speech.
    """
    QUALITY = 0      # Least aggressive, may include some noise as speech
    LOW_BITRATE = 1  # Balanced
    AGGRESSIVE = 2   # More aggressive
    VERY_AGGRESSIVE = 3  # Most aggressive, may miss some soft speech


@dataclass
class VADResult:
    """Result from VAD analysis."""
    is_speech: bool          # True if human speech detected
    rms_level: float         # RMS energy level
    confidence: str          # "high", "medium", "low"
    method: str              # "webrtc", "rms_fallback"


class EnhancedVAD:
    """
    Enhanced Voice Activity Detection combining WebRTC VAD with RMS analysis.
    
    WebRTC VAD is specifically trained to detect human speech and ignore:
    - Coughs and sneezes
    - Background noise
    - Music
    - Environmental sounds
    
    Parameters:
        sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
        aggressiveness: 0-3, higher = stricter speech detection
        rms_threshold: Fallback RMS threshold if webrtcvad unavailable
        frame_duration_ms: Frame size (10, 20, or 30 ms)
    """
    
    def __init__(
        self,
        sample_rate: int = 8000,
        aggressiveness: int = VADAggressiveness.AGGRESSIVE,
        rms_threshold: float = 800,
        frame_duration_ms: int = 20
    ):
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.frame_duration_ms = frame_duration_ms
        
        # Calculate expected frame size
        self.frame_size_samples = int(sample_rate * frame_duration_ms / 1000)
        self.frame_size_bytes = self.frame_size_samples * 2  # 16-bit = 2 bytes
        
        # Initialize WebRTC VAD if available
        self._vad: Optional[webrtcvad.Vad] = None
        if WEBRTC_AVAILABLE:
            try:
                self._vad = webrtcvad.Vad(aggressiveness)
                logger.info(
                    f"WebRTC VAD initialized: rate={sample_rate}Hz, "
                    f"aggressiveness={aggressiveness}, frame={frame_duration_ms}ms"
                )
            except Exception as e:
                logger.error(f"Failed to initialize WebRTC VAD: {e}")
                self._vad = None
        
        # Statistics
        self._total_frames = 0
        self._speech_frames = 0
    
    def process_audio(self, pcm_audio: bytes) -> VADResult:
        """
        Analyze audio frame for speech.
        
        Args:
            pcm_audio: Raw PCM audio bytes (16-bit signed, mono)
            
        Returns:
            VADResult with speech detection info
        """
        self._total_frames += 1
        
        # Calculate RMS for energy level
        rms = self._calculate_rms(pcm_audio)
        
        # Use WebRTC VAD if available
        if self._vad is not None:
            try:
                is_speech = self._vad.is_speech(pcm_audio, self.sample_rate)
                
                if is_speech:
                    self._speech_frames += 1
                
                # Determine confidence based on RMS + VAD agreement
                if is_speech and rms > self.rms_threshold:
                    confidence = "high"
                elif is_speech:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                return VADResult(
                    is_speech=is_speech,
                    rms_level=rms,
                    confidence=confidence,
                    method="webrtc"
                )
                
            except Exception as e:
                # Fall through to RMS fallback
                logger.debug(f"WebRTC VAD error: {e}")
        
        # Fallback to RMS-only detection
        is_speech = rms > self.rms_threshold
        if is_speech:
            self._speech_frames += 1
            
        return VADResult(
            is_speech=is_speech,
            rms_level=rms,
            confidence="medium" if is_speech else "low",
            method="rms_fallback"
        )
    
    def is_speech(self, pcm_audio: bytes) -> bool:
        """
        Simple boolean check for speech.
        
        Args:
            pcm_audio: Raw PCM audio bytes
            
        Returns:
            True if speech detected
        """
        return self.process_audio(pcm_audio).is_speech
    
    def _calculate_rms(self, pcm_audio: bytes) -> float:
        """Calculate RMS energy level."""
        if len(pcm_audio) < 2:
            return 0.0
        
        try:
            import audioop
            return float(audioop.rms(pcm_audio, 2))
        except Exception:
            # Manual calculation fallback
            try:
                count = len(pcm_audio) // 2
                samples = struct.unpack(f'{count}h', pcm_audio)
                sum_squares = sum(s * s for s in samples)
                return (sum_squares / count) ** 0.5
            except Exception:
                return 0.0
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "total_frames": self._total_frames,
            "speech_frames": self._speech_frames,
            "speech_ratio": (
                self._speech_frames / self._total_frames 
                if self._total_frames > 0 else 0
            ),
            "method": "webrtc" if self._vad else "rms_fallback",
            "webrtc_available": WEBRTC_AVAILABLE
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_frames = 0
        self._speech_frames = 0


class SmartBargeInDetector:
    """
    Intelligent barge-in detection using WebRTC VAD.
    
    Combines:
    1. WebRTC VAD for speech/not-speech classification
    2. RMS threshold for energy check
    3. Confirmation frames for stability
    
    This filters out coughs, sneezes, and noise while still
    being responsive to actual speech.
    """
    
    def __init__(
        self,
        sample_rate: int = 8000,
        aggressiveness: int = VADAggressiveness.AGGRESSIVE,
        rms_threshold: float = 800,
        confirmation_frames: int = 2,
        cooldown_ms: int = 500,
        grace_period_ms: int = 0
    ):
        self.vad = EnhancedVAD(
            sample_rate=sample_rate,
            aggressiveness=aggressiveness,
            rms_threshold=rms_threshold
        )
        self.confirmation_frames = confirmation_frames
        self.cooldown_ms = cooldown_ms
        self.grace_period_ms = grace_period_ms
        
        # State
        self._consecutive_speech = 0
        self._is_monitoring = False
        self._last_trigger_time: Optional[float] = None
        self._monitoring_start_time: Optional[float] = None
        
        logger.info(
            f"SmartBargeInDetector: confirm_frames={confirmation_frames}, "
            f"cooldown={cooldown_ms}ms, grace_period={grace_period_ms}ms"
        )
    
    def start_monitoring(self) -> None:
        """Start monitoring for barge-in (when agent starts speaking)."""
        self._is_monitoring = True
        self._consecutive_speech = 0
        import time
        self._monitoring_start_time = time.time()
        logger.debug("🔊 Started barge-in monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring (when agent stops speaking)."""
        self._is_monitoring = False
        self._consecutive_speech = 0
        logger.debug("🔇 Stopped barge-in monitoring")
    
    def check_barge_in(self, pcm_audio: bytes) -> Tuple[bool, VADResult]:
        """
        Check if audio triggers barge-in.
        
        Args:
            pcm_audio: Raw PCM audio bytes
            
        Returns:
            Tuple of (is_barge_in, vad_result)
        """
        if not self._is_monitoring:
            return False, VADResult(False, 0, "low", "disabled")
        
        import time
        now = time.time()
        
        # Check grace period (ignore first X ms of monitoring)
        if self._monitoring_start_time is not None:
            elapsed_ms = (now - self._monitoring_start_time) * 1000
            if elapsed_ms < self.grace_period_ms:
                return False, VADResult(False, 0, "low", "grace_period")
        
        # Check cooldown
        if self._last_trigger_time is not None:
            elapsed_ms = (now - self._last_trigger_time) * 1000
            if elapsed_ms < self.cooldown_ms:
                return False, VADResult(False, 0, "low", "cooldown")
        
        # Analyze audio
        result = self.vad.process_audio(pcm_audio)
        
        # 1. SPEECH-BASED BARGE-IN (Conservative)
        # Must be both VAD speech AND above RMS energy threshold
        if result.is_speech and result.rms_level >= self.vad.rms_threshold:
            self._consecutive_speech += 1
        # 2. EMERGENCY STOP (Loud sounds)
        # If sound is very loud, trigger faster even without VAD confirmation
        elif result.rms_level >= self.vad.rms_threshold * 2:
            self._consecutive_speech += 1
        else:
            # Reset counter on silence/low noise
            self._consecutive_speech = 0

        if self._consecutive_speech >= self.confirmation_frames:
            # Barge-in confirmed!
            self._last_trigger_time = time.time()
            self._is_monitoring = False  # One-shot
            
            trigger_method = "speech" if result.is_speech else "loud_sound"
            logger.info(
                f"🚨 BARGE-IN detected! "
                f"(method={trigger_method}, RMS={result.rms_level:.0f}, "
                f"confidence={result.confidence})"
            )
            return True, result
        
        return False, result
    
    @property
    def is_monitoring(self) -> bool:
        return self._is_monitoring
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            **self.vad.get_stats(),
            "consecutive_speech": self._consecutive_speech,
            "is_monitoring": self._is_monitoring
        }
