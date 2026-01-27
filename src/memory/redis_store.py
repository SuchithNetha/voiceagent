"""
Redis-based Persistent Memory Store for Sarah Voice Agent.

Provides cross-session memory persistence using Redis:
- User identification via phone number or session ID
- Conversation history storage
- User preference persistence
- Fast retrieval for returning users

Cloud-ready with support for:
- Redis Cloud (Redis Labs)
- AWS ElastiCache
- Azure Cache for Redis
- Upstash (Serverless Redis)
"""

import json
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import asdict
import asyncio

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.memory.models import (
    SessionMemory, 
    UserPreferences, 
    ConversationTurn,
    PropertyInterest,
    ConversationSnapshot,
    IntentType
)
from src.utils.logger import setup_logging

logger = setup_logging("Memory-Redis")


class RedisConfig:
    """Redis connection configuration."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        ssl: bool = False,
        # Cloud provider URLs (use these for managed Redis)
        url: Optional[str] = None,  # redis://user:password@host:port/db
        # Key prefixes
        prefix: str = "sarah:",
        # TTL settings
        session_ttl_hours: int = 24,       # Session expires after 24 hours
        user_profile_ttl_days: int = 90,   # User profiles expire after 90 days
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.ssl = ssl
        self.url = url
        self.prefix = prefix
        self.session_ttl_hours = session_ttl_hours
        self.user_profile_ttl_days = user_profile_ttl_days
    
    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create config from environment variables."""
        import os
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", "0")),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            url=os.getenv("REDIS_URL"),  # Full connection URL
        )


class RedisMemoryStore:
    """
    Redis-backed persistent memory store.
    
    Key Structure:
    - sarah:user:{user_id}:profile     - User preferences and profile
    - sarah:user:{user_id}:sessions    - List of session IDs
    - sarah:user:{user_id}:last_summary - Last conversation summary
    - sarah:session:{session_id}       - Full session data
    - sarah:phone:{phone_hash}         - Maps phone number to user_id
    
    This enables:
    1. User recognition when they call back
    2. Preference persistence across sessions
    3. Conversation continuity
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis library not installed. "
                "Install with: pip install redis[hiredis]"
            )
        
        self.config = config or RedisConfig()
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        
    async def connect(self) -> bool:
        """Establish connection to Redis with retry logic."""
        return await self._connect_with_retry()
    
    async def _connect_with_retry(self, is_reconnect: bool = False) -> bool:
        """Connect to Redis with exponential backoff retry."""
        max_attempts = self._max_reconnect_attempts if is_reconnect else 1
        
        for attempt in range(max_attempts):
            try:
                if self.config.url:
                    # Use connection URL (for cloud providers)
                    self._client = redis.from_url(
                        self.config.url,
                        decode_responses=True
                    )
                else:
                    # Use individual parameters
                    self._client = redis.Redis(
                        host=self.config.host,
                        port=self.config.port,
                        password=self.config.password,
                        db=self.config.db,
                        ssl=self.config.ssl,
                        decode_responses=True
                    )
                
                # Test connection
                await self._client.ping()
                self._connected = True
                self._reconnect_attempts = 0
                logger.info(f"✅ Connected to Redis at {self.config.host}:{self.config.port}")
                return True
                
            except Exception as e:
                self._reconnect_attempts += 1
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                
                if attempt < max_attempts - 1:
                    logger.warning(f"⚠️ Redis connection attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Redis connection failed after {max_attempts} attempts: {e}")
                    self._connected = False
                    return False
        
        return False
    
    async def ensure_connected(self) -> bool:
        """Ensure Redis is connected, attempt reconnect if not."""
        if self._connected:
            try:
                await self._client.ping()
                return True
            except Exception:
                logger.warning("⚠️ Redis connection lost, attempting reconnect...")
                self._connected = False
        
        return await self._connect_with_retry(is_reconnect=True)
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info("🔌 Redis connection closed")
    
    def _key(self, *parts: str) -> str:
        """Generate Redis key with prefix."""
        return self.config.prefix + ":".join(parts)
    
    def _hash_phone(self, phone_number: str) -> str:
        """Hash phone number for privacy."""
        return hashlib.sha256(phone_number.encode()).hexdigest()[:16]
    
    # --- USER IDENTIFICATION ---
    
    async def get_or_create_user_id(
        self, 
        phone_number: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Get existing user ID or create a new one.
        
        Priority:
        1. Look up by phone number (best for returning callers)
        2. Use session_id if no phone
        3. Generate new user_id
        """
        if not self._connected:
            await self.connect()
        
        # Try to find existing user by phone
        if phone_number:
            phone_hash = self._hash_phone(phone_number)
            phone_key = self._key("phone", phone_hash)
            
            existing_user_id = await self._client.get(phone_key)
            if existing_user_id:
                logger.info(f"🔁 Returning user identified: {existing_user_id[:8]}...")
                return existing_user_id
            
            # New user - create mapping
            import uuid
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            await self._client.set(phone_key, user_id)
            
            # Store reverse mapping for debugging
            await self._client.hset(
                self._key("user", user_id, "meta"),
                "phone_hash", phone_hash
            )
            
            logger.info(f"🆕 New user created: {user_id[:8]}...")
            return user_id
        
        # Fallback to session-based ID
        if session_id:
            return f"session_{session_id}"
        
        # Last resort - anonymous
        import uuid
        return f"anon_{uuid.uuid4().hex[:8]}"
    
    # --- SESSION MANAGEMENT ---
    
    async def save_session(self, session: SessionMemory, user_id: str) -> bool:
        """
        Save session to Redis.
        
        Args:
            session: Session memory to save
            user_id: User identifier
        """
        # Try to reconnect if disconnected
        if not await self.ensure_connected():
            return False
        
        try:
            session_key = self._key("session", session.session_id)
            
            # Serialize session
            session_data = {
                "session_id": session.session_id,
                "user_id": user_id,
                "started_at": session.started_at.isoformat(),
                "turns": [self._serialize_turn(t) for t in session.turns],
                "preferences": session.preferences.to_dict(),
                "properties_discussed": [p.to_dict() for p in session.properties_discussed],
                "running_summary": session.running_summary,
                "last_summary_turn": session.last_summary_turn,
                "current_intent": session.current_intent.value if session.current_intent else None,
                "last_property_shown": session.last_property_shown,
            }
            
            # Store with TTL
            ttl_seconds = self.config.session_ttl_hours * 3600
            await self._client.setex(
                session_key,
                ttl_seconds,
                json.dumps(session_data)
            )
            
            # Add to user's session list
            user_sessions_key = self._key("user", user_id, "sessions")
            await self._client.lpush(user_sessions_key, session.session_id)
            await self._client.ltrim(user_sessions_key, 0, 9)  # Keep last 10 sessions
            
            logger.debug(f"💾 Session saved: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[SessionMemory]:
        """Load session from Redis."""
        if not await self.ensure_connected():
            return None
        
        try:
            session_key = self._key("session", session_id)
            data = await self._client.get(session_key)
            
            if not data:
                return None
            
            session_data = json.loads(data)
            return self._deserialize_session(session_data)
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    # --- USER PROFILE ---
    
    async def save_user_profile(
        self, 
        user_id: str, 
        preferences: UserPreferences,
        last_summary: Optional[str] = None
    ) -> bool:
        """Save user profile for long-term persistence."""
        if not await self.ensure_connected():
            return False
        
        try:
            profile_key = self._key("user", user_id, "profile")
            
            profile_data = {
                "user_id": user_id,
                "preferences": preferences.to_dict(),
                "last_summary": last_summary,
                "updated_at": datetime.now().isoformat(),
            }
            
            ttl_seconds = self.config.user_profile_ttl_days * 86400
            await self._client.setex(
                profile_key,
                ttl_seconds,
                json.dumps(profile_data)
            )
            
            logger.debug(f"💾 User profile saved: {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user profile: {e}")
            return False
    
    async def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user profile from Redis."""
        if not self._connected:
            return None
        
        try:
            profile_key = self._key("user", user_id, "profile")
            data = await self._client.get(profile_key)
            
            if not data:
                return None
            
            return json.loads(data)
            
        except Exception as e:
            logger.error(f"Failed to load user profile: {e}")
            return None
    
    async def get_user_context(self, user_id: str) -> Optional[str]:
        """
        Get context string for returning user.
        
        Returns a summary of previous interactions for the agent to use.
        """
        profile = await self.load_user_profile(user_id)
        
        if not profile:
            return None
        
        parts = []
        
        # Add last conversation summary
        if profile.get("last_summary"):
            parts.append(f"Previous conversation: {profile['last_summary']}")
        
        # Add preferences
        prefs = profile.get("preferences", {})
        if prefs.get("preferred_locations"):
            locs = ", ".join(prefs["preferred_locations"][:3])
            parts.append(f"Interested in: {locs}")
        if prefs.get("max_price"):
            parts.append(f"Budget: up to €{prefs['max_price']:,.0f}")
        if prefs.get("style_preferences"):
            styles = list(prefs["style_preferences"].keys())[:2]
            parts.append(f"Prefers: {', '.join(styles)}")
        
        return " | ".join(parts) if parts else None
    
    # --- RECENT SESSIONS ---
    
    async def get_recent_sessions(
        self, 
        user_id: str, 
        limit: int = 5
    ) -> List[ConversationSnapshot]:
        """Get summaries of recent sessions for a user."""
        if not self._connected:
            return []
        
        try:
            user_sessions_key = self._key("user", user_id, "sessions")
            session_ids = await self._client.lrange(user_sessions_key, 0, limit - 1)
            
            snapshots = []
            for sid in session_ids:
                session = await self.load_session(sid)
                if session:
                    snapshots.append(ConversationSnapshot.from_session(session))
            
            return snapshots
            
        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []
    
    # --- SERIALIZATION HELPERS ---
    
    def _serialize_turn(self, turn: ConversationTurn) -> dict:
        """Serialize a conversation turn."""
        return {
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat(),
            "intent": turn.intent.value if turn.intent else None,
            "entities": turn.entities,
            "tool_used": turn.tool_used,
            "confidence": turn.confidence,
        }
    
    def _deserialize_session(self, data: dict) -> SessionMemory:
        """Deserialize session from Redis data."""
        # Reconstruct turns
        turns = []
        for t in data.get("turns", []):
            turns.append(ConversationTurn(
                role=t["role"],
                content=t["content"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                intent=IntentType(t["intent"]) if t.get("intent") else None,
                entities=t.get("entities", {}),
                tool_used=t.get("tool_used"),
                confidence=t.get("confidence", 1.0)
            ))
        
        # Reconstruct preferences
        pref_data = data.get("preferences", {})
        preferences = UserPreferences(
            preferred_locations=pref_data.get("preferred_locations", []),
            excluded_locations=pref_data.get("excluded_locations", []),
            max_price=pref_data.get("max_price"),
            min_price=pref_data.get("min_price"),
            price_flexibility=pref_data.get("price_flexibility", "moderate"),
            min_rooms=pref_data.get("min_rooms"),
            max_rooms=pref_data.get("max_rooms"),
            min_baths=pref_data.get("min_baths"),
            min_sqft=pref_data.get("min_sqft"),
            style_preferences=pref_data.get("style_preferences", {}),
            must_have=pref_data.get("must_have", []),
            nice_to_have=pref_data.get("nice_to_have", []),
        )
        
        # Reconstruct properties
        properties = []
        for p in data.get("properties_discussed", []):
            properties.append(PropertyInterest(
                property_id=p["property_id"],
                location=p["location"],
                price=p.get("price"),
                rooms=p.get("rooms"),
                times_discussed=p.get("times_discussed", 1),
                user_reaction=p.get("user_reaction"),
                notes=p.get("notes", "")
            ))
        
        return SessionMemory(
            session_id=data["session_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            turns=turns,
            preferences=preferences,
            properties_discussed=properties,
            running_summary=data.get("running_summary"),
            last_summary_turn=data.get("last_summary_turn", 0),
            current_intent=IntentType(data["current_intent"]) if data.get("current_intent") else None,
            last_property_shown=data.get("last_property_shown"),
        )
    
    # --- HEALTH CHECK ---
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health."""
        if not self._connected:
            return {"status": "disconnected", "healthy": False}
        
        try:
            start = datetime.now()
            await self._client.ping()
            latency_ms = (datetime.now() - start).total_seconds() * 1000
            
            info = await self._client.info("memory")
            
            return {
                "status": "connected",
                "healthy": True,
                "latency_ms": round(latency_ms, 2),
                "used_memory": info.get("used_memory_human", "unknown"),
            }
        except Exception as e:
            return {"status": "error", "healthy": False, "error": str(e)}


# --- SINGLETON ACCESS ---
_redis_store: Optional[RedisMemoryStore] = None


async def get_redis_store() -> RedisMemoryStore:
    """Get or create Redis memory store singleton."""
    global _redis_store
    
    if _redis_store is None:
        config = RedisConfig.from_env()
        _redis_store = RedisMemoryStore(config)
        await _redis_store.connect()
    
    return _redis_store


async def init_redis_memory() -> bool:
    """Initialize Redis memory store at startup."""
    try:
        store = await get_redis_store()
        health = await store.health_check()
        return health.get("healthy", False)
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        return False
