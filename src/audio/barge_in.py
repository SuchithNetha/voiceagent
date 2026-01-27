"""
Barge-in Handler for Soft Full-Duplex Conversations.

This module handles user interruptions during TTS playback, enabling
more natural conversations where users can interrupt the agent.

Hard Duplex: Agent speaks until finished, user must wait
Soft Duplex: User can interrupt at any time, agent gracefully stops
"""

import asyncio
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from src.audio.rms_monitor import RMSMonitor
from src.utils.logger import setup_logging

logger = setup_logging("Audio-BargeIn")


class ConversationState(Enum):
    """Current state of the conversation."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class BargeInEvent:
    """Details of a barge-in event."""
    timestamp: datetime
    rms_level: float
    speech_position_ms: int  # How far into TTS playback
    reason: str  # "user_speech", "timeout", etc.


class BargeInHandler:
    """
    Handles user interruptions during TTS playback.
    
    Enables soft full-duplex by:
    1. Monitoring incoming audio while speaking
    2. Detecting user speech above threshold
    3. Signaling TTS to stop gracefully
    4. Transitioning back to listening mode
    
    This creates a more natural, conversational experience where the
    user doesn't have to wait for the agent to finish speaking.
    """
    
    def __init__(
        self,
        rms_monitor: RMSMonitor,
        barge_in_threshold: float = 1500,
        confirmation_frames: int = 3,  # ~60ms of speech to confirm
        cooldown_ms: int = 500  # Prevent rapid re-triggering
    ):
        """
        Initialize barge-in handler.
        
        Args:
            rms_monitor: RMS monitor instance for audio analysis
            barge_in_threshold: RMS level that triggers barge-in
            confirmation_frames: Consecutive frames needed to confirm barge-in
            cooldown_ms: Minimum time between barge-in events
        """
        self.rms_monitor = rms_monitor
        self.barge_in_threshold = barge_in_threshold
        self.confirmation_frames = confirmation_frames
        self.cooldown_ms = cooldown_ms
        
        # State
        self._state = ConversationState.IDLE
        self._is_playing = False
        self._barge_in_detected = asyncio.Event()
        self._stop_playback = asyncio.Event()
        
        # Tracking
        self._consecutive_speech_frames = 0
        self._playback_start_time: Optional[datetime] = None
        self._last_barge_in: Optional[datetime] = None
        self._barge_in_callback: Optional[Callable] = None
        
        # Statistics
        self._total_barge_ins = 0
        self._total_playbacks = 0
        
        logger.info(
            f"BargeInHandler initialized: "
            f"threshold={barge_in_threshold}, "
            f"confirm_frames={confirmation_frames}"
        )
    
    @property
    def state(self) -> ConversationState:
        """Current conversation state."""
        return self._state
    
    def set_barge_in_callback(self, callback: Callable[[BargeInEvent], None]) -> None:
        """Set callback for when barge-in occurs."""
        self._barge_in_callback = callback
    
    def start_playback(self) -> None:
        """Signal that TTS playback has started."""
        self._is_playing = True
        self._state = ConversationState.SPEAKING
        self._playback_start_time = datetime.now()
        self._barge_in_detected.clear()
        self._stop_playback.clear()
        self._consecutive_speech_frames = 0
        self._total_playbacks += 1
        logger.debug("🔊 Playback started, monitoring for barge-in")
    
    def stop_playback(self) -> None:
        """Signal that TTS playback has ended normally."""
        self._is_playing = False
        self._state = ConversationState.LISTENING
        self._playback_start_time = None
        logger.debug("🔇 Playback ended normally")
    
    def process_incoming_audio(self, pcm_audio: bytes) -> bool:
        """
        Process incoming audio during playback to detect barge-in.
        
        Args:
            pcm_audio: Raw PCM audio from user
            
        Returns:
            True if barge-in was detected
        """
        if not self._is_playing:
            return False
        
        # Check cooldown
        if self._last_barge_in:
            elapsed = (datetime.now() - self._last_barge_in).total_seconds() * 1000
            if elapsed < self.cooldown_ms:
                return False
        
        # Analyze audio
        analysis = self.rms_monitor.process_audio(pcm_audio)
        
        # Check for potential barge-in
        if analysis.current_rms > self.barge_in_threshold:
            self._consecutive_speech_frames += 1
            
            if self._consecutive_speech_frames >= self.confirmation_frames:
                # Barge-in confirmed!
                self._trigger_barge_in(analysis.current_rms)
                return True
        else:
            # Reset counter if below threshold
            self._consecutive_speech_frames = 0
        
        return False
    
    def _trigger_barge_in(self, rms_level: float) -> None:
        """Handle confirmed barge-in."""
        now = datetime.now()
        self._last_barge_in = now
        self._total_barge_ins += 1
        
        # Calculate position in playback
        position_ms = 0
        if self._playback_start_time:
            position_ms = int((now - self._playback_start_time).total_seconds() * 1000)
        
        # Create event
        event = BargeInEvent(
            timestamp=now,
            rms_level=rms_level,
            speech_position_ms=position_ms,
            reason="user_speech"
        )
        
        # Update state
        self._state = ConversationState.INTERRUPTED
        self._barge_in_detected.set()
        self._stop_playback.set()
        self._is_playing = False
        
        logger.info(
            f"🚨 BARGE-IN at {position_ms}ms into playback "
            f"(RMS: {rms_level:.0f})"
        )
        
        # Trigger callback if registered
        if self._barge_in_callback:
            try:
                self._barge_in_callback(event)
            except Exception as e:
                logger.error(f"Barge-in callback error: {e}")
    
    async def monitor_playback(
        self, 
        audio_stream: AsyncGenerator[bytes, None],
        on_barge_in: Optional[Callable] = None
    ) -> None:
        """
        Async task to monitor incoming audio during playback.
        
        Run this alongside TTS output to enable barge-in detection.
        
        Args:
            audio_stream: Async generator yielding audio chunks
            on_barge_in: Optional callback when barge-in detected
        """
        logger.debug("Starting barge-in monitor task")
        
        async for audio_chunk in audio_stream:
            if not self._is_playing:
                break
            
            if self.process_incoming_audio(audio_chunk):
                if on_barge_in:
                    await on_barge_in() if asyncio.iscoroutinefunction(on_barge_in) else on_barge_in()
                break
    
    async def wait_for_barge_in(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for barge-in event.
        
        Args:
            timeout: Maximum seconds to wait, None for indefinite
            
        Returns:
            True if barge-in occurred, False if timeout
        """
        try:
            await asyncio.wait_for(
                self._barge_in_detected.wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def should_stop_playback(self) -> bool:
        """Check if playback should be stopped."""
        return self._stop_playback.is_set()
    
    def reset(self) -> None:
        """Reset handler for new conversation."""
        self._state = ConversationState.IDLE
        self._is_playing = False
        self._barge_in_detected.clear()
        self._stop_playback.clear()
        self._consecutive_speech_frames = 0
        self._playback_start_time = None
        self._last_barge_in = None
        logger.debug("BargeInHandler reset")
    
    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            "state": self._state.value,
            "is_playing": self._is_playing,
            "total_playbacks": self._total_playbacks,
            "total_barge_ins": self._total_barge_ins,
            "barge_in_rate": (
                self._total_barge_ins / self._total_playbacks 
                if self._total_playbacks > 0 else 0
            )
        }


class SoftDuplexController:
    """
    High-level controller for soft full-duplex conversation management.
    
    Coordinates between:
    - Audio input monitoring
    - TTS output streaming
    - Barge-in detection
    - State transitions
    """
    
    def __init__(
        self,
        rms_monitor: RMSMonitor,
        barge_in_handler: BargeInHandler
    ):
        self.rms_monitor = rms_monitor
        self.barge_in_handler = barge_in_handler
        
        self._playback_task: Optional[asyncio.Task] = None
        
    async def speak_with_barge_in_support(
        self,
        tts_generator: AsyncGenerator,
        input_audio_stream: AsyncGenerator[bytes, None],
        output_callback: Callable
    ) -> bool:
        """
        Stream TTS output while monitoring for barge-in.
        
        Args:
            tts_generator: Generator yielding TTS audio chunks
            input_audio_stream: Stream of incoming audio for monitoring
            output_callback: Function to send audio to user
            
        Returns:
            True if completed normally, False if interrupted
        """
        self.barge_in_handler.start_playback()
        
        try:
            # Start monitoring in background
            monitor_task = asyncio.create_task(
                self.barge_in_handler.monitor_playback(input_audio_stream)
            )
            
            # Stream TTS output
            async for audio_chunk in tts_generator:
                if self.barge_in_handler.should_stop_playback():
                    logger.info("⚡ Stopping TTS due to barge-in")
                    return False
                
                await output_callback(audio_chunk)
            
            return True
            
        finally:
            self.barge_in_handler.stop_playback()
            if 'monitor_task' in locals():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
