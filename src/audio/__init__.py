"""
Audio processing utilities for Sarah Voice Agent.

This package contains:
- RMS monitoring for voice activity detection
- Adaptive silence detection
- Barge-in handling for full-duplex conversations
- WebRTC VAD for intelligent speech detection
"""

from src.audio.rms_monitor import RMSMonitor
from src.audio.adaptive_vad import AdaptiveSilenceDetector
from src.audio.barge_in import BargeInHandler
from src.audio.webrtc_vad import SmartBargeInDetector, EnhancedVAD, WEBRTC_AVAILABLE

__all__ = [
    "RMSMonitor", 
    "AdaptiveSilenceDetector", 
    "BargeInHandler",
    "SmartBargeInDetector",
    "EnhancedVAD",
    "WEBRTC_AVAILABLE"
]
