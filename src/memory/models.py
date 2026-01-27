"""
Memory Models for Sarah Voice Agent.

Data structures for conversation memory:
- Working memory (current context)
- Episodic memory (session history)
- Semantic memory (user preferences, property interests)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class IntentType(Enum):
    """Types of user intents detected in conversation."""
    SEARCH = "search"           # Looking for properties
    DETAILS = "details"         # Asking about specific property
    COMPARE = "compare"         # Comparing properties
    NEGOTIATE = "negotiate"     # Discussing price/terms
    SCHEDULE = "schedule"       # Scheduling viewing
    GENERAL = "general"         # General conversation
    GREETING = "greeting"       # Hello/goodbye
    CLARIFY = "clarify"         # Asking for clarification


@dataclass
class ConversationTurn:
    """Single turn in conversation with metadata."""
    role: str  # "user" or "assistant" or "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[IntentType] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    tool_used: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent.value if self.intent else None,
            "entities": self.entities,
            "tool_used": self.tool_used,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            intent=IntentType(data["intent"]) if data.get("intent") else None,
            entities=data.get("entities", {}),
            tool_used=data.get("tool_used"),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class PropertyInterest:
    """Record of a property discussed in conversation."""
    property_id: str
    location: str
    price: Optional[float] = None
    rooms: Optional[int] = None
    first_mentioned: datetime = field(default_factory=datetime.now)
    times_discussed: int = 1
    user_reaction: Optional[str] = None  # "positive", "negative", "neutral"
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "location": self.location,
            "price": self.price,
            "rooms": self.rooms,
            "first_mentioned": self.first_mentioned.isoformat(),
            "times_discussed": self.times_discussed,
            "user_reaction": self.user_reaction,
            "notes": self.notes
        }


@dataclass
class UserPreferences:
    """Extracted user preferences during conversation."""
    # Location preferences
    preferred_locations: List[str] = field(default_factory=list)
    excluded_locations: List[str] = field(default_factory=list)
    
    # Price preferences
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    price_flexibility: str = "moderate"  # "strict", "moderate", "flexible"
    
    # Property characteristics
    min_rooms: Optional[int] = None
    max_rooms: Optional[int] = None
    min_baths: Optional[int] = None
    min_sqft: Optional[float] = None
    
    # Style preferences (for weighted search)
    style_preferences: Dict[str, float] = field(default_factory=dict)
    # e.g., {"modern": 0.8, "charming": 0.3, "luxury": 0.5}
    
    # Requirements vs nice-to-haves
    must_have: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)
    
    # Derived confidence scores
    preference_confidence: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "preferred_locations": self.preferred_locations,
            "excluded_locations": self.excluded_locations,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "price_flexibility": self.price_flexibility,
            "min_rooms": self.min_rooms,
            "max_rooms": self.max_rooms,
            "min_baths": self.min_baths,
            "min_sqft": self.min_sqft,
            "style_preferences": self.style_preferences,
            "must_have": self.must_have,
            "nice_to_have": self.nice_to_have,
            "preference_confidence": self.preference_confidence
        }
    
    def update_from_entity(self, entity_type: str, value: Any, confidence: float = 0.8):
        """Update preference from extracted entity."""
        if entity_type == "location":
            if value not in self.preferred_locations:
                self.preferred_locations.append(value)
                self.preference_confidence["location"] = confidence
                
        elif entity_type == "max_price":
            if self.max_price is None or value < self.max_price:
                self.max_price = value
                self.preference_confidence["price"] = confidence
                
        elif entity_type == "rooms":
            self.min_rooms = value
            self.preference_confidence["rooms"] = confidence
            
        elif entity_type == "style":
            self.style_preferences[value] = self.style_preferences.get(value, 0) + 0.2
            # Cap at 1.0
            self.style_preferences[value] = min(1.0, self.style_preferences[value])


@dataclass
class SessionMemory:
    """
    Memory for current conversation session.
    
    This tracks:
    - All conversation turns
    - Extracted user preferences
    - Properties discussed
    - Current conversation state
    """
    session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    
    # Conversation history
    turns: List[ConversationTurn] = field(default_factory=list)
    
    # Extracted information
    preferences: UserPreferences = field(default_factory=UserPreferences)
    properties_discussed: List[PropertyInterest] = field(default_factory=list)
    
    # Summarized context
    running_summary: Optional[str] = None
    last_summary_turn: int = 0
    
    # Current state
    current_intent: Optional[IntentType] = None
    pending_question: Optional[str] = None
    last_property_shown: Optional[str] = None
    
    def add_turn(
        self, 
        role: str, 
        content: str, 
        intent: Optional[IntentType] = None,
        entities: Optional[Dict] = None
    ) -> ConversationTurn:
        """Add a new conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            entities=entities or {}
        )
        self.turns.append(turn)
        
        if role == "user" and intent:
            self.current_intent = intent
        
        return turn
    
    def add_property_interest(
        self,
        property_id: str,
        location: str,
        price: Optional[float] = None,
        rooms: Optional[int] = None
    ) -> PropertyInterest:
        """Record a property of interest."""
        # Check if already discussed
        for prop in self.properties_discussed:
            if prop.property_id == property_id:
                prop.times_discussed += 1
                return prop
        
        interest = PropertyInterest(
            property_id=property_id,
            location=location,
            price=price,
            rooms=rooms
        )
        self.properties_discussed.append(interest)
        self.last_property_shown = property_id
        return interest
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the n most recent turns."""
        return self.turns[-n:] if self.turns else []
    
    def get_user_messages(self) -> List[str]:
        """Get all user messages."""
        return [turn.content for turn in self.turns if turn.role == "user"]
    
    def get_assistant_messages(self) -> List[str]:
        """Get all assistant messages."""
        return [turn.content for turn in self.turns if turn.role == "assistant"]
    
    def needs_summary(self, threshold: int = 10) -> bool:
        """Check if conversation needs summarization."""
        return len(self.turns) - self.last_summary_turn >= threshold
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "turns": [t.to_dict() for t in self.turns],
            "preferences": self.preferences.to_dict(),
            "properties_discussed": [p.to_dict() for p in self.properties_discussed],
            "running_summary": self.running_summary,
            "last_summary_turn": self.last_summary_turn,
            "current_intent": self.current_intent.value if self.current_intent else None,
            "pending_question": self.pending_question,
            "last_property_shown": self.last_property_shown
        }
    
    def get_context_string(self, max_turns: int = 5) -> str:
        """Generate context string for LLM prompts."""
        parts = []
        
        # Add summary if available
        if self.running_summary:
            parts.append(f"[Session Summary: {self.running_summary}]")
        
        # Add preference summary if we have some
        if self.preferences.preferred_locations:
            parts.append(
                f"[User interested in: {', '.join(self.preferences.preferred_locations)}]"
            )
        if self.preferences.max_price:
            parts.append(f"[Budget: up to {self.preferences.max_price:,.0f} EUR]")
        
        # Add recent turns
        recent = self.get_recent_turns(max_turns)
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Sarah"
            parts.append(f"{prefix}: {turn.content}")
        
        return "\n".join(parts)


@dataclass  
class ConversationSnapshot:
    """Lightweight snapshot for quick context retrieval."""
    session_id: str
    turn_count: int
    summary: str
    key_preferences: Dict[str, Any]
    last_property: Optional[str]
    timestamp: datetime
    
    @classmethod
    def from_session(cls, session: SessionMemory) -> "ConversationSnapshot":
        """Create snapshot from full session."""
        key_prefs = {}
        if session.preferences.preferred_locations:
            key_prefs["locations"] = session.preferences.preferred_locations[:3]
        if session.preferences.max_price:
            key_prefs["max_price"] = session.preferences.max_price
        if session.preferences.style_preferences:
            key_prefs["style"] = list(session.preferences.style_preferences.keys())[:3]
        
        return cls(
            session_id=session.session_id,
            turn_count=len(session.turns),
            summary=session.running_summary or "No summary yet",
            key_preferences=key_prefs,
            last_property=session.last_property_shown,
            timestamp=datetime.now()
        )
