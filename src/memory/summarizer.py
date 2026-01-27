"""
Memory Summarizer for Sarah Voice Agent.

Generates concise summaries of conversations for:
- Context window management
- Long-term storage
- Cross-session continuity
"""

import asyncio
from typing import Optional, List
from datetime import datetime

from src.memory.models import SessionMemory, UserPreferences, IntentType
from src.utils.logger import setup_logging

logger = setup_logging("Memory-Summarizer")


class MemorySummarizer:
    """
    Generates concise summaries of real estate conversations.
    
    Uses LLM to create summaries that capture:
    - User's property preferences
    - Properties discussed and reactions
    - Outstanding questions/requests
    """
    
    SUMMARY_PROMPT = """You are summarizing a real estate conversation between a user and Sarah, a real estate agent in Madrid.

Create a brief summary (under 80 words) that captures:
1. User's property preferences (location, price, style)
2. Properties shown and user's reactions
3. Any outstanding requests or questions

Be specific about preferences mentioned. Focus on actionable information.

Conversation:
{conversation}

Summary:"""

    PREFERENCE_EXTRACTION_PROMPT = """Extract user preferences from this real estate conversation.

Return JSON with these fields (use null if not mentioned):
- locations: list of mentioned locations/neighborhoods
- max_price: maximum budget mentioned (number only)
- min_rooms: minimum rooms requested
- style_keywords: list of style words (modern, charming, luxury, cozy, etc.)
- must_have: list of requirements mentioned
- nice_to_have: list of preferences that aren't requirements

Conversation:
{conversation}

JSON:"""

    def __init__(self, llm=None):
        """
        Initialize summarizer.
        
        Args:
            llm: LangChain LLM instance for summarization
        """
        self._llm = llm
        self._summary_cache = {}  # session_id -> (turn_count, summary)
        
    def set_llm(self, llm):
        """Set LLM for summarization."""
        self._llm = llm
        
    async def summarize_session(
        self, 
        session: SessionMemory,
        force: bool = False
    ) -> str:
        """
        Generate summary of conversation session.
        
        Args:
            session: Session memory to summarize
            force: If True, regenerate even if cached
            
        Returns:
            Summary string
        """
        if not self._llm:
            return self._fallback_summary(session)
        
        # Check cache
        cache_key = session.session_id
        if cache_key in self._summary_cache and not force:
            cached_turns, cached_summary = self._summary_cache[cache_key]
            if cached_turns == len(session.turns):
                return cached_summary
        
        # Build conversation text
        conversation_text = self._format_conversation(session)
        
        try:
            # Use LLM for summary with timeout to prevent hanging
            prompt = self.SUMMARY_PROMPT.format(conversation=conversation_text)
            try:
                response = await asyncio.wait_for(
                    self._llm.ainvoke(prompt),
                    timeout=10.0  # 10 second timeout
                )
                summary = response.content.strip()
            except asyncio.TimeoutError:
                logger.warning("LLM summarization timed out, using fallback")
                return self._fallback_summary(session)
            
            # Cache result
            self._summary_cache[cache_key] = (len(session.turns), summary)
            
            # Update session
            session.running_summary = summary
            session.last_summary_turn = len(session.turns)
            
            logger.info(f"Generated summary for session {session.session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._fallback_summary(session)
    
    def _fallback_summary(self, session: SessionMemory) -> str:
        """Generate summary without LLM using rule-based extraction."""
        parts = []
        
        # Summarize preferences
        prefs = session.preferences
        if prefs.preferred_locations:
            parts.append(f"Looking in: {', '.join(prefs.preferred_locations[:3])}")
        if prefs.max_price:
            parts.append(f"Budget: up to {prefs.max_price:,.0f}€")
        if prefs.min_rooms:
            parts.append(f"Needs: {prefs.min_rooms}+ rooms")
        if prefs.style_preferences:
            top_styles = sorted(
                prefs.style_preferences.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:2]
            styles = ", ".join([s[0] for s in top_styles])
            parts.append(f"Prefers: {styles}")
        
        # Summarize properties
        if session.properties_discussed:
            prop_count = len(session.properties_discussed)
            parts.append(f"Shown: {prop_count} properties")
            
            # Note any positive reactions
            positive = [p for p in session.properties_discussed if p.user_reaction == "positive"]
            if positive:
                locations = ", ".join([p.location for p in positive[:2]])
                parts.append(f"Liked: {locations}")
        
        return " | ".join(parts) if parts else "New conversation"
    
    def _format_conversation(
        self, 
        session: SessionMemory,
        max_turns: int = 15
    ) -> str:
        """Format conversation for LLM input."""
        lines = []
        
        # Include running summary if we have one (for incremental updates)
        if session.running_summary and len(session.turns) > max_turns:
            lines.append(f"[Previous: {session.running_summary}]")
            # Only include turns after last summary
            turns = session.turns[session.last_summary_turn:][-max_turns:]
        else:
            turns = session.turns[-max_turns:]
        
        for turn in turns:
            role = "User" if turn.role == "user" else "Sarah"
            lines.append(f"{role}: {turn.content}")
        
        return "\n".join(lines)
    
    async def extract_preferences(
        self, 
        session: SessionMemory
    ) -> UserPreferences:
        """
        Extract user preferences from conversation using LLM.
        
        Args:
            session: Session to extract from
            
        Returns:
            Updated UserPreferences
        """
        if not self._llm:
            return self._extract_preferences_rules(session)
        
        conversation_text = self._format_conversation(session)
        
        try:
            prompt = self.PREFERENCE_EXTRACTION_PROMPT.format(
                conversation=conversation_text
            )
            
            # Call LLM with timeout
            try:
                response = await asyncio.wait_for(
                    self._llm.ainvoke(prompt),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("LLM preference extraction timed out")
                return self._extract_preferences_rules(session)
            
            # Parse JSON response
            import json
            # Try to extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            # Update preferences
            prefs = session.preferences
            if data.get("locations"):
                for loc in data["locations"]:
                    if loc not in prefs.preferred_locations:
                        prefs.preferred_locations.append(loc)
            if data.get("max_price"):
                prefs.max_price = float(data["max_price"])
            if data.get("min_rooms"):
                prefs.min_rooms = int(data["min_rooms"])
            if data.get("style_keywords"):
                for style in data["style_keywords"]:
                    prefs.style_preferences[style.lower()] = (
                        prefs.style_preferences.get(style.lower(), 0) + 0.3
                    )
            if data.get("must_have"):
                prefs.must_have.extend(data["must_have"])
            if data.get("nice_to_have"):
                prefs.nice_to_have.extend(data["nice_to_have"])
                
            logger.debug(f"Extracted preferences: {prefs.to_dict()}")
            return prefs
            
        except Exception as e:
            logger.warning(f"LLM preference extraction failed: {e}")
            return self._extract_preferences_rules(session)
    
    def _extract_preferences_rules(
        self, 
        session: SessionMemory
    ) -> UserPreferences:
        """Rule-based preference extraction as fallback."""
        prefs = session.preferences
        
        # Simple keyword extraction from user messages
        style_keywords = {
            "modern": ["modern", "contemporary", "new", "renovated"],
            "charming": ["charming", "cozy", "warm", "character", "quaint"],
            "luxury": ["luxury", "luxurious", "high-end", "premium", "exclusive"],
            "spacious": ["spacious", "large", "big", "roomy"],
            "bright": ["bright", "light", "sunny", "airy"]
        }
        
        location_keywords = [
            "salamanca", "chamberí", "retiro", "centro", "argüelles",
            "chamartín", "malasaña", "chueca", "hortaleza", "moncloa"
        ]
        
        for turn in session.turns:
            if turn.role != "user":
                continue
                
            content_lower = turn.content.lower()
            
            # Extract styles
            for style, keywords in style_keywords.items():
                for kw in keywords:
                    if kw in content_lower:
                        prefs.style_preferences[style] = min(
                            1.0, prefs.style_preferences.get(style, 0) + 0.2
                        )
            
            # Extract locations
            for loc in location_keywords:
                if loc in content_lower and loc.title() not in prefs.preferred_locations:
                    prefs.preferred_locations.append(loc.title())
            
            # Extract price (simple pattern matching)
            import re
            price_matches = re.findall(r'(\d{3,}(?:[,.\s]\d{3})*)\s*(?:euros?|€|eur)', content_lower)
            if price_matches:
                try:
                    # Take the last mentioned price
                    price_str = price_matches[-1].replace(",", "").replace(".", "").replace(" ", "")
                    price = float(price_str)
                    if 50000 < price < 50000000:  # Reasonable range
                        prefs.max_price = price
                except ValueError:
                    pass
            
            # Extract room counts
            room_matches = re.findall(r'(\d+)\s*(?:bedrooms?|rooms?|habitacion)', content_lower)
            if room_matches:
                prefs.min_rooms = int(room_matches[-1])
        
        return prefs
    
    def summarize_for_greeting(self, session: SessionMemory) -> Optional[str]:
        """Generate a brief summary for session start greeting."""
        if not session.turns:
            return None
        
        prefs = session.preferences
        parts = []
        
        if prefs.preferred_locations:
            parts.append(f"properties in {prefs.preferred_locations[0]}")
        if prefs.max_price:
            parts.append(f"around {prefs.max_price/1000:.0f}K euros")
        if prefs.min_rooms:
            parts.append(f"with {prefs.min_rooms} bedrooms")
        
        if parts:
            return f"looking for {', '.join(parts)}"
        return None
