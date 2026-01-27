# Voice Agent Enhancement Plan

## Overview
This document outlines the implementation plan for enhancing the Sarah Voice Agent with the following features:

1. **Turn-based STT → Streaming STT** - Real-time transcription
2. **Superlinked Schema Enhancement** - Weighted subjective descriptors + soft constraints
3. **Adaptive Silence Detection** - 800ms → 600ms or adaptive threshold
4. **RMS Monitoring + Soft Duplex Barge-in** - Full duplex interruption handling
5. **Memory Summaries** - Conversation context and summarization

---

## 1. Streaming STT Implementation

### Current State
- Using `ReplyOnPause` from FastRTC which buffers complete audio and transcribes after silence detection
- Turn-based: User speaks → Silence detected → Transcribe → AI responds

### Target State
- Real-time streaming transcription using `RealtimeSTT` or Whisper streaming mode
- Progressive transcription updates while user is speaking
- Early intent detection for faster response times

### Implementation Steps
1. Create `src/models/streaming_stt.py` with streaming Whisper implementation
2. Implement audio chunk processor with rolling buffer
3. Add partial transcription callback for early processing
4. Integrate with VAD for intelligent speech boundary detection

### Key Files to Modify
- `src/models/stt.py` → Add streaming variant
- `src/main.py` → Replace `ReplyOnPause` handler
- `src/telephony.py` → Update media stream handler

---

## 2. Superlinked Schema Enhancement

### Current State
```python
class Property(sl.Schema):
    id: sl.IdField
    description: sl.String
    baths: sl.Float
    rooms: sl.Integer
    sqft: sl.Float
    location: sl.String
    price: sl.Float
```

Only uses `TextSimilaritySpace` (description) and `NumberSpace` (price with MINIMUM mode).

### Target State
Add weighted subjective descriptors and soft constraints:

```python
class Property(sl.Schema):
    id: sl.IdField
    description: sl.String
    baths: sl.Float
    rooms: sl.Integer
    sqft: sl.Float
    location: sl.String
    price: sl.Float
    # NEW: Subjective descriptors (computed from descriptions or added to CSV)
    charm_score: sl.Float        # 0-1 scale for "charming" properties
    modern_score: sl.Float       # 0-1 scale for "modern" properties
    luxury_score: sl.Float       # 0-1 scale for luxury level
    cozy_score: sl.Float         # 0-1 scale for coziness
```

### Implementation Steps
1. **Data Enrichment**: Create a script to analyze descriptions and compute subjective scores
2. **Schema Update**: Add weighted subjective descriptor fields to Superlinked schema
3. **Multi-Space Index**: Add separate embedding spaces for each subjective trait
4. **Soft Constraints**: Use dynamic weighting instead of hard filters
5. **Query Enhancement**: Implement preference-based ranking with weight parameters

### New Query Structure
```python
query = (
    sl.Query(property_index)
    .find(property_schema)
    .similar(description_space, sl.Param("description_query"), weight=0.4)
    .similar(charm_space, sl.Param("charm_preference"), weight=0.2)
    .similar(modern_space, sl.Param("modern_preference"), weight=0.2)
    # Soft constraint: prefer, don't filter
    .similar(price_space, sl.Param("price_target"), weight=0.2)
    .limit(3)  # Return top 3 for variety
)
```

---

## 3. Adaptive Silence Detection (800ms → 600ms / Adaptive)

### Current State (telephony.py:64)
```python
SILENCE_LIMIT = 800  # 800ms of silence counts as 'finished'
```

### Target State
- Reduce baseline to 600ms for more responsive feel
- Implement adaptive threshold based on:
  - Recent RMS levels (speech energy)
  - Speaking pattern (fast/slow talker)
  - Conversation context (end of question vs mid-sentence)

### Implementation
```python
class AdaptiveSilenceDetector:
    def __init__(self):
        self.base_silence_ms = 600  # Down from 800ms
        self.min_silence_ms = 400   # Minimum for fast talkers
        self.max_silence_ms = 800   # Maximum for slow talkers
        self.recent_rms_values = deque(maxlen=50)
        self.speech_rate_estimate = 1.0  # Words per second estimate
        
    def get_dynamic_threshold(self, current_rms: float, is_question: bool) -> int:
        """Calculate dynamic silence threshold based on context."""
        # Adjust based on speech energy
        avg_rms = sum(self.recent_rms_values) / len(self.recent_rms_values)
        energy_factor = current_rms / avg_rms if avg_rms > 0 else 1.0
        
        # Questions may need longer pause for user to formulate answer
        context_bonus = 100 if is_question else 0
        
        threshold = self.base_silence_ms * (1 / energy_factor) + context_bonus
        return max(self.min_silence_ms, min(self.max_silence_ms, int(threshold)))
```

---

## 4. RMS Monitoring + Soft Full Duplex Barge-in

### Current State
- Hard Half-Duplex: Sarah speaks until finished, then listens
- No interruption detection during TTS playback
- RMS checked only for silence detection

### Target State
- Soft Full Duplex: Monitor user audio while Sarah speaks
- Detect barge-in (user interruption) and gracefully stop TTS
- Smooth transition from speaking to listening

### Implementation Components

#### 4.1 RMS Monitor Service
```python
class RMSMonitor:
    """Continuous RMS monitoring for barge-in and VAD."""
    
    def __init__(self, threshold: float = 1000, window_size: int = 10):
        self.threshold = threshold
        self.rms_history = deque(maxlen=window_size)
        self.is_user_speaking = False
        self.callbacks = []
        
    def process_audio(self, pcm_audio: bytes) -> dict:
        rms = audioop.rms(pcm_audio, 2)
        self.rms_history.append(rms)
        
        # Detect speech onset
        avg_rms = sum(self.rms_history) / len(self.rms_history)
        was_speaking = self.is_user_speaking
        self.is_user_speaking = avg_rms > self.threshold
        
        return {
            "current_rms": rms,
            "average_rms": avg_rms,
            "is_speaking": self.is_user_speaking,
            "speech_started": self.is_user_speaking and not was_speaking,
            "speech_ended": not self.is_user_speaking and was_speaking,
        }
```

#### 4.2 Barge-in Handler
```python
class BargeInHandler:
    """Handles user interruptions during TTS playback."""
    
    def __init__(self, rms_monitor: RMSMonitor):
        self.rms_monitor = rms_monitor
        self.is_playing = False
        self.barge_in_detected = asyncio.Event()
        self.barge_in_threshold = 1500  # Higher than silence threshold
        
    async def monitor_for_barge_in(self, audio_stream: AsyncGenerator):
        """Monitor incoming audio while playing TTS."""
        consecutive_speech_frames = 0
        REQUIRED_FRAMES = 3  # ~60ms of speech to confirm barge-in
        
        async for audio_chunk in audio_stream:
            if not self.is_playing:
                break
                
            result = self.rms_monitor.process_audio(audio_chunk)
            if result["current_rms"] > self.barge_in_threshold:
                consecutive_speech_frames += 1
                if consecutive_speech_frames >= REQUIRED_FRAMES:
                    self.barge_in_detected.set()
                    break
            else:
                consecutive_speech_frames = 0
```

#### 4.3 Soft Duplex Flow
```
User speaking → [Detection] → [Transcribe streaming] → [AI thinks]
                                                              ↓
                                                        [TTS starts]
                                                              ↓
                              [Monitor for barge-in] ← [Audio plays]
                                      ↓
                        [User interrupts = stop TTS] → [Listen mode]
```

---

## 5. Memory Summaries Design

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Memory System                             │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Working Memory│  │Episodic Memory│  │Semantic Memory│   │
│  │ (Current)     │  │ (Session)     │  │ (Long-term)   │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│         ↓                  ↓                   ↓            │
│  Current context    Session summary      User preferences   │
│  Last 5 turns       Property interests   Style patterns     │
│  Active intent      Shown properties     Past sessions      │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

#### 5.1 Memory Models
```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None
    entities: Dict = field(default_factory=dict)

@dataclass
class SessionMemory:
    """Memory for current session."""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    preferences_extracted: Dict = field(default_factory=dict)
    properties_discussed: List[str] = field(default_factory=list)
    session_summary: Optional[str] = None
    
@dataclass
class UserProfile:
    """Long-term user preferences."""
    user_id: str
    preferred_locations: List[str] = field(default_factory=list)
    price_range: Optional[tuple] = None
    style_preferences: List[str] = field(default_factory=list)  # modern, charming, etc.
    interaction_count: int = 0
    last_session_summary: Optional[str] = None
```

#### 5.2 Summary Generator
```python
class MemorySummarizer:
    """Generates concise summaries of conversations."""
    
    SUMMARY_PROMPT = """
    Summarize this real estate conversation concisely:
    - User's property preferences
    - Properties discussed
    - Outstanding questions/requests
    
    Keep it under 100 words. Be specific about preferences.
    
    Conversation:
    {conversation}
    """
    
    async def summarize_session(self, session: SessionMemory) -> str:
        """Generate summary of session for storage."""
        conversation_text = "\n".join([
            f"{turn.role}: {turn.content}" 
            for turn in session.turns[-10:]  # Last 10 turns
        ])
        
        # Use LLM to generate summary
        summary = await self.llm.ainvoke(
            self.SUMMARY_PROMPT.format(conversation=conversation_text)
        )
        return summary.content
        
    def extract_preferences(self, session: SessionMemory) -> Dict:
        """Extract user preferences from conversation."""
        preferences = {
            "locations": [],
            "price_range": None,
            "rooms_min": None,
            "style": [],
            "priorities": [],
        }
        
        for turn in session.turns:
            if turn.role == "user":
                # NLP extraction of preferences
                # ... extract locations, price mentions, style words
                pass
        
        return preferences
```

#### 5.3 Context Window Manager
```python
class ContextWindowManager:
    """Manages what context to include in LLM prompts."""
    
    def __init__(self, max_turns: int = 5, max_tokens: int = 2000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        
    def build_context(
        self, 
        session: SessionMemory, 
        user_profile: Optional[UserProfile] = None
    ) -> str:
        """Build context string for LLM prompt."""
        
        context_parts = []
        
        # Add session summary if available
        if session.session_summary:
            context_parts.append(f"Session context: {session.session_summary}")
        
        # Add user profile summary if returning user
        if user_profile and user_profile.last_session_summary:
            context_parts.append(f"Previous session: {user_profile.last_session_summary}")
            if user_profile.preferred_locations:
                context_parts.append(
                    f"Known preferences: {', '.join(user_profile.preferred_locations)}"
                )
        
        # Add recent turns (most recent first, up to limit)
        recent_turns = session.turns[-self.max_turns:]
        for turn in recent_turns:
            context_parts.append(f"{turn.role}: {turn.content}")
        
        return "\n".join(context_parts)
```

---

## File Changes Summary

### New Files to Create
1. `src/models/streaming_stt.py` - Streaming STT implementation
2. `src/audio/rms_monitor.py` - RMS monitoring service
3. `src/audio/barge_in.py` - Barge-in detection handler  
4. `src/audio/adaptive_vad.py` - Adaptive silence detection
5. `src/memory/models.py` - Memory data models
6. `src/memory/summarizer.py` - Session summarization
7. `src/memory/context.py` - Context window management
8. `src/tools/data_enrichment.py` - Script to compute subjective scores for properties

### Files to Modify
1. `src/main.py` - Integrate streaming STT, memory, barge-in
2. `src/telephony.py` - Update VAD, add barge-in monitoring
3. `src/tools/property_search.py` - Enhanced Superlinked schema with weights
4. `src/models/llm.py` - Update system prompt with memory context
5. `src/config.py` - Add new configuration parameters
6. `data/properties.csv` - Add subjective score columns (computed)

---

## Implementation Priority

### Phase 1: Quick Wins (Immediate Impact)
1. ✅ Reduce silence threshold 800ms → 600ms
2. ✅ Add basic RMS monitoring/logging

### Phase 2: Response Quality
3. Enhanced Superlinked schema with soft constraints
4. Memory summaries for context

### Phase 3: Advanced UX
5. Streaming STT
6. Full duplex barge-in handling
7. Adaptive silence detection

---

## Configuration Parameters

Add to `config.py`:
```python
# Audio/VAD
SILENCE_THRESHOLD_MS: int = 600
SILENCE_THRESHOLD_MIN_MS: int = 400
SILENCE_THRESHOLD_MAX_MS: int = 800
RMS_SILENCE_THRESHOLD: int = 1000
RMS_BARGE_IN_THRESHOLD: int = 1500

# Memory
MEMORY_MAX_TURNS: int = 5
MEMORY_SUMMARY_INTERVAL: int = 10  # Summarize every 10 turns
MEMORY_PERSISTENCE: bool = False  # Enable file-based storage

# Search
SEARCH_DESCRIPTION_WEIGHT: float = 0.4
SEARCH_CHARM_WEIGHT: float = 0.15
SEARCH_MODERN_WEIGHT: float = 0.15
SEARCH_PRICE_WEIGHT: float = 0.3
```
