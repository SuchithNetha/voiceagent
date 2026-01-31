"""
Context Window Manager for Sarah Voice Agent.

Manages what context to include in LLM prompts to:
- Stay within token limits
- Provide relevant conversation history
- Include user preferences efficiently
"""

from typing import Optional, List
from dataclasses import dataclass

from src.memory.models import SessionMemory, ConversationTurn
from src.utils.logger import setup_logging

logger = setup_logging("Memory-Context")


@dataclass
class ContextConfig:
    """Configuration for context window management."""
    max_turns: int = 2  # Only keep last 2 turns to reduce repetition
    max_tokens_estimate: int = 1500  # Smaller context window
    include_summary: bool = False  # Don't include - causes repetition
    include_preferences: bool = True  # Keep preferences (useful)
    include_properties: bool = False  # Don't repeat property info
    chars_per_token: int = 4


class ContextWindowManager:
    """
    Manages context building for LLM prompts.
    
    Ensures the agent has relevant context without exceeding
    token limits or including irrelevant information.
    
    Context priority (highest to lowest):
    1. Current user message (always included)
    2. Conversation summary (if available)
    3. User preferences
    4. Recent turns (up to limit)
    5. Properties discussed
    """
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize context manager.
        
        Args:
            config: Context configuration, uses defaults if None
        """
        self.config = config or ContextConfig()
        
    def build_context(
        self,
        session: SessionMemory,
        current_message: Optional[str] = None,
        include_system: bool = False
    ) -> str:
        """
        Build context string for LLM prompt.
        
        Args:
            session: Current session memory
            current_message: The user's current message (if not yet in session)
            include_system: Whether to include system instructions
            
        Returns:
            Formatted context string
        """
        parts = []
        token_budget = self.config.max_tokens_estimate
        
        # 1. Add session summary if available
        if self.config.include_summary and session.running_summary:
            summary_text = f"[Conversation so far: {session.running_summary}]"
            if self._estimate_tokens(summary_text) < token_budget * 0.2:
                parts.append(summary_text)
                token_budget -= self._estimate_tokens(summary_text)
        
        # 2. Add key preferences
        if self.config.include_preferences:
            pref_text = self._format_preferences(session)
            if pref_text and self._estimate_tokens(pref_text) < token_budget * 0.15:
                parts.append(pref_text)
                token_budget -= self._estimate_tokens(pref_text)
        
        # 3. Add properties discussed
        if self.config.include_properties and session.properties_discussed:
            prop_text = self._format_properties(session)
            if prop_text and self._estimate_tokens(prop_text) < token_budget * 0.15:
                parts.append(prop_text)
                token_budget -= self._estimate_tokens(prop_text)
        
        # 4. Add recent turns (fit as many as budget allows)
        turns_text = self._format_recent_turns(session, token_budget * 0.5)
        if turns_text:
            parts.append(turns_text)
        
        return "\n\n".join(parts)
    
    def _format_preferences(self, session: SessionMemory) -> Optional[str]:
        """Format user preferences for context."""
        prefs = session.preferences
        items = []
        
        if prefs.preferred_locations:
            locs = ", ".join(prefs.preferred_locations[:3])
            items.append(f"Locations: {locs}")
        
        if prefs.max_price:
            items.append(f"Budget: up to €{prefs.max_price:,.0f}")
        
        if prefs.min_rooms:
            items.append(f"Bedrooms: {prefs.min_rooms}+")
        
        if prefs.style_preferences:
            # Get top 3 styles
            sorted_styles = sorted(
                prefs.style_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            styles = ", ".join([s[0] for s in sorted_styles if s[1] > 0.3])
            if styles:
                items.append(f"Style: {styles}")
        
        if not items:
            return None
        
        return "[User Preferences: " + " | ".join(items) + "]"
    
    def _format_properties(self, session: SessionMemory) -> Optional[str]:
        """Format discussed properties for context."""
        if not session.properties_discussed:
            return None
        
        # Only include last 3 properties
        recent = session.properties_discussed[-3:]
        summaries = []
        
        for prop in recent:
            summary = f"{prop.location}"
            if prop.price:
                summary += f" (€{prop.price:,.0f})"
            if prop.user_reaction:
                summary += f" - {prop.user_reaction}"
            summaries.append(summary)
        
        return "[Properties shown: " + "; ".join(summaries) + "]"
    
    def _format_recent_turns(
        self, 
        session: SessionMemory, 
        max_tokens: float
    ) -> Optional[str]:
        """Format recent conversation turns within token budget."""
        if not session.turns:
            return None
        
        lines = []
        tokens_used = 0
        
        # Work backwards from most recent
        for turn in reversed(session.turns[-self.config.max_turns:]):
            role = "User" if turn.role == "user" else "Arya"
            line = f"{role}: {turn.content}"
            line_tokens = self._estimate_tokens(line)
            
            if tokens_used + line_tokens > max_tokens:
                break
            
            lines.insert(0, line)  # Insert at beginning to maintain order
            tokens_used += line_tokens
        
        if not lines:
            return None
        
        return "[Recent conversation:]\n" + "\n".join(lines)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return len(text) // self.config.chars_per_token
    
    def build_search_context(self, session: SessionMemory) -> dict:
        """
        Build context specifically for property search.
        
        Returns dict with search parameters derived from context.
        """
        prefs = session.preferences
        
        context = {
            "description_query": None,
            "location_filter": None,
            "price_target": None,
            "rooms_min": None,
            "style_weights": {},
        }
        
        # Build description query from recent user messages
        recent_user = [
            turn.content for turn in session.turns[-3:] 
            if turn.role == "user"
        ]
        if recent_user:
            context["description_query"] = " ".join(recent_user)
        
        # Add location filter if strongly preferred
        if prefs.preferred_locations:
            context["location_filter"] = prefs.preferred_locations[0]
        
        # Add price target
        if prefs.max_price:
            context["price_target"] = prefs.max_price
        
        # Add room minimum
        if prefs.min_rooms:
            context["rooms_min"] = prefs.min_rooms
        
        # Add style weights for weighted search
        if prefs.style_preferences:
            context["style_weights"] = dict(prefs.style_preferences)
        
        return context
    
    def get_agent_system_context(self, session: SessionMemory) -> str:
        """
        Generate context to inject into agent system prompt.
        
        This provides the agent with relevant user information
        without cluttering the main conversation.
        """
        lines = []
        
        # Add known information about the user
        if session.preferences.preferred_locations:
            lines.append(
                f"The user is interested in: {', '.join(session.preferences.preferred_locations[:3])}"
            )
        
        if session.preferences.max_price:
            lines.append(
                f"Their budget is up to €{session.preferences.max_price:,.0f}"
            )
        
        if session.preferences.style_preferences:
            top_styles = sorted(
                session.preferences.style_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            if top_styles:
                styles = ", ".join([s[0] for s in top_styles])
                lines.append(f"They prefer {styles} properties")
        
        if session.properties_discussed:
            count = len(session.properties_discussed)
            lines.append(f"You have shown them {count} properties so far")
            
            # Note if they liked any
            liked = [p for p in session.properties_discussed if p.user_reaction == "positive"]
            if liked:
                lines.append(
                    f"They responded positively to properties in: {', '.join([p.location for p in liked[:2]])}"
                )
        
        if session.pending_question:
            lines.append(f"Outstanding question: {session.pending_question}")
        
        if not lines:
            return ""
        
        return "\n[User Context]\n" + "\n".join(f"- {line}" for line in lines)
