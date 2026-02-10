"""
Persistent Session Manager for Sarah Voice Agent.

Orchestrates memory persistence across sessions:
- Identifies returning users via phone number
- Loads previous context and preferences
- Saves session state for continuity
- Manages memory lifecycle

Works with:
- Redis for persistent storage
- In-memory fallback when Redis unavailable
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

from src.memory.models import SessionMemory, UserPreferences
from src.memory.summarizer import MemorySummarizer
from src.memory.context import ContextWindowManager
from src.utils.logger import setup_logging
from src.config import get_config

logger = setup_logging("Memory-SessionManager")


@dataclass
class UserContext:
    """Context for a returning user."""
    user_id: str
    is_returning: bool
    greeting_context: Optional[str]  # For personalized greeting
    preferences: Optional[UserPreferences]
    last_session_summary: Optional[str]
    was_interrupted: bool = False


class PersistentSessionManager:
    """
    Manages session lifecycle with persistent storage.
    
    Flow for incoming call:
    1. Identify user (by phone or generate new)
    2. Load previous context if returning user
    3. Create new session with context
    4. Periodically save session state
    5. Summarize and persist on session end
    
    Usage:
        manager = PersistentSessionManager()
        await manager.start()
        
        # When call comes in
        context = await manager.start_session(phone_number="+1234567890")
        if context.is_returning:
            greeting = f"Welcome back! Last time we discussed {context.greeting_context}"
        
        # During conversation
        manager.add_turn(session_id, role="user", content="...")
        
        # When call ends
        await manager.end_session(session_id)
    """
    
    def __init__(
        self,
        use_redis: bool = True,
        auto_save_interval: int = 30,  # seconds
        summarize_on_end: bool = True
    ):
        self.use_redis = use_redis
        self.auto_save_interval = auto_save_interval
        self.summarize_on_end = summarize_on_end
        
        # Storage
        self._redis_store = None
        self._local_sessions: Dict[str, SessionMemory] = {}
        self._session_users: Dict[str, str] = {}  # session_id -> user_id
        
        # Services
        self._summarizer = MemorySummarizer()
        self._context_manager = ContextWindowManager()
        
        # Background tasks
        self._save_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self, llm=None) -> bool:
        """
        Initialize the session manager.
        
        Args:
            llm: Optional LLM for summarization
        """
        logger.info("🚀 Starting Persistent Session Manager...")
        
        # Set up summarizer with LLM
        if llm:
            self._summarizer.set_llm(llm)
        
        # Connect to Redis if enabled
        if self.use_redis:
            try:
                from src.memory.redis_store import get_redis_store
                self._redis_store = await get_redis_store()
                logger.info("✅ Redis storage connected")
            except ImportError:
                logger.warning("⚠️ Redis not available, using in-memory only")
                self._redis_store = None
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}, using in-memory only")
                self._redis_store = None
        
        # Start auto-save task
        self._running = True
        self._save_task = asyncio.create_task(self._auto_save_loop())
        
        # Start cleanup task to prevent memory leaks
        self._cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        
        logger.info("✅ Session Manager ready")
        return True
    
    async def _session_cleanup_loop(self):
        """Background task to clean up stale sessions (prevents memory leaks)."""
        MAX_SESSION_AGE_HOURS = 2
        MAX_LOCAL_SESSIONS = 100  # Safety limit
        
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                now = datetime.now()
                stale_sessions = []
                
                for session_id, session in list(self._local_sessions.items()):
                    age_hours = (now - session.started_at).total_seconds() / 3600
                    if age_hours > MAX_SESSION_AGE_HOURS:
                        stale_sessions.append(session_id)
                
                # Clean up stale sessions
                for session_id in stale_sessions:
                    logger.warning(f"🧹 Cleaning up stale session: {session_id}")
                    await self.end_session(session_id)
                
                # Safety limit check
                if len(self._local_sessions) > MAX_LOCAL_SESSIONS:
                    oldest = sorted(
                        self._local_sessions.items(),
                        key=lambda x: x[1].started_at
                    )[:len(self._local_sessions) - MAX_LOCAL_SESSIONS]
                    for session_id, _ in oldest:
                        logger.warning(f"🧹 Evicting oldest session: {session_id}")
                        await self.end_session(session_id)
                
                if stale_sessions:
                    logger.info(f"🧹 Cleaned {len(stale_sessions)} stale sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    async def stop(self):
        """Shutdown session manager gracefully."""
        logger.info("🛑 Stopping Session Manager...")
        
        self._running = False
        
        # Cancel background tasks
        for task in [self._save_task, getattr(self, '_cleanup_task', None)]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save all active sessions
        for session_id in list(self._local_sessions.keys()):
            await self.end_session(session_id)
        
        # Disconnect Redis
        if self._redis_store:
            await self._redis_store.disconnect()
        
        logger.info("✅ Session Manager stopped")
    
    async def start_session(
        self,
        session_id: str,
        phone_number: Optional[str] = None
    ) -> UserContext:
        """
        Start a new session, identifying user if possible.
        
        Args:
            session_id: Unique session identifier (e.g., Twilio stream SID)
            phone_number: Caller's phone number for identification
            
        Returns:
            UserContext with user info and previous context
        """
        user_id = None
        is_returning = False
        greeting_context = None
        preferences = None
        last_summary = None
        was_interrupted = False
        
        # Try to identify user via Redis
        if self._redis_store and phone_number:
            try:
                user_id = await self._redis_store.get_or_create_user_id(
                    phone_number=phone_number
                )
                
                # Load previous context
                profile = await self._redis_store.load_user_profile(user_id)
                if profile:
                    is_returning = True
                    last_summary = profile.get("last_summary")
                    was_interrupted = not profile.get("last_session_graceful", True)
                    
                    # Reconstruct preferences
                    pref_data = profile.get("preferences", {})
                    if pref_data:
                        preferences = UserPreferences(
                            preferred_locations=pref_data.get("preferred_locations", []),
                            max_price=pref_data.get("max_price"),
                            style_preferences=pref_data.get("style_preferences", {}),
                        )
                    
                    # Build greeting context
                    greeting_context = await self._redis_store.get_user_context(user_id)
                    
                    logger.info(f"🔁 Returning user: {user_id[:8]}... - {greeting_context}")
                else:
                    logger.info(f"🆕 New user: {user_id[:8]}...")
                    
            except Exception as e:
                logger.error(f"Error identifying user: {e}")
        
        # Fallback user ID
        if not user_id:
            user_id = f"session_{session_id}"
        
        # Create new session
        session = SessionMemory(
            session_id=session_id,
            started_at=datetime.now(),
            preferences=preferences or UserPreferences()
        )
        
        # Store locally
        self._local_sessions[session_id] = session
        self._session_users[session_id] = user_id
        
        logger.info(f"📞 Session started: {session_id}")
        
        return UserContext(
            user_id=user_id,
            is_returning=is_returning,
            greeting_context=greeting_context,
            preferences=preferences,
            last_session_summary=last_summary,
            was_interrupted=was_interrupted
        )
    
    async def end_session(self, session_id: str) -> bool:
        """
        End session and persist state.
        
        This:
        1. Generates conversation summary
        2. Saves session to Redis
        3. Updates user profile
        4. Cleans up local state
        """
        session = self._local_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        user_id = self._session_users.get(session_id)
        
        try:
            # Mark session as ended gracefully
            session.ended_gracefully = True
            
            # Generate summary if we have turns
            if self.summarize_on_end and len(session.turns) > 0:
                summary = await self._summarizer.summarize_session(session)
                session.running_summary = summary
                logger.info(f"📝 Session summary: {summary[:100]}...")
            
            # Extract and update preferences
            if len(session.turns) > 2:
                await self._summarizer.extract_preferences(session)
            
            # Persist to Redis
            if self._redis_store and user_id:
                # Save session
                await self._redis_store.save_session(session, user_id)
                
                # Update user profile
                await self._redis_store.save_user_profile(
                    user_id=user_id,
                    preferences=session.preferences,
                    last_summary=session.running_summary,
                    last_session_graceful=session.ended_gracefully
                )
                
                logger.info(f"💾 Session persisted for user {user_id[:8]}...")
            
            # Clean up local state
            del self._local_sessions[session_id]
            if session_id in self._session_users:
                del self._session_users[session_id]
            
            logger.info(f"✅ Session ended: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending session: {e}", exc_info=True)
            return False
    
    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get active session by ID."""
        return self._local_sessions.get(session_id)
    
    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        **kwargs
    ) -> bool:
        """Add a conversation turn to session."""
        session = self._local_sessions.get(session_id)
        if not session:
            return False
        
        session.add_turn(role=role, content=content, **kwargs)
        return True
    
    def get_context_for_llm(self, session_id: str) -> str:
        """Get context string for LLM prompt."""
        session = self._local_sessions.get(session_id)
        if not session:
            return ""
        
        return self._context_manager.build_context(session)
    
    async def _auto_save_loop(self):
        """Background task to periodically save sessions."""
        while self._running:
            try:
                await asyncio.sleep(self.auto_save_interval)
                
                if not self._redis_store:
                    continue
                
                # Save all active sessions
                for session_id, session in self._local_sessions.items():
                    user_id = self._session_users.get(session_id)
                    if user_id:
                        await self._redis_store.save_session(session, user_id)
                
                if self._local_sessions:
                    logger.debug(f"💾 Auto-saved {len(self._local_sessions)} sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
    
    # --- CONVENIENCE METHODS ---
    
    def get_user_greeting(self, context: UserContext) -> str:
        """Generate personalized greeting for user."""
        if context.is_returning:
            if context.was_interrupted:
                return (
                    "Hey there! I'm so sorry, it seems our last call was cut short unexpectedly. "
                    "I still remember everything we talked about, though! Should we pick up where we left off with your property search?"
                )
            
            # Warm, natural greeting for returning users
            return "Hey there! Welcome back. It's so good to hear from you again. How can I help you continue your property search today?"
        else:
            # Premium, inviting greeting for new users
            return (
                "Hi! This is Arya. I'm so excited to help you find your dream home in Madrid. "
                "What kind of vibe are you looking for? A modern apartment, or something with a bit more history?"
            )
    
    async def get_returning_user_summary(self, session_id: str) -> Optional[str]:
        """Get summary of returning user's previous interactions."""
        user_id = self._session_users.get(session_id)
        if not user_id or not self._redis_store:
            return None
        
        return await self._redis_store.get_user_context(user_id)

    async def list_users(self) -> List[Dict[str, Any]]:
        """List all users in the system."""
        if not self._redis_store:
            return []
        return await self._redis_store.list_all_user_profiles()

    async def list_historical_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List historical sessions from Redis."""
        if not self._redis_store:
            return []
        return await self._redis_store.list_all_sessions(limit=limit)

    # --- AUTHENTICATION ---
    
    async def create_user(self, username: str, password_plain: str, role: str = "user", approved: bool = False):
        if not self._redis_store: return False
        
        # Check if this is the configured Super Admin
        config = get_config()
        if username == config.SUPER_ADMIN_USERNAME:
            role = "super_admin"
            approved = True
            
        return await self._redis_store.save_user(username, password_plain, role, approved)

    async def authenticate_user(self, username: str, password_plain: str) -> Optional[Dict[str, Any]]:
        if not self._redis_store: return None
        from src.utils.auth import verify_password
        user = await self._redis_store.get_user_auth(username)
        if user and verify_password(password_plain, user["password_hash"]):
            # HQ OVERRIDE: Only the designated Commander (Ghost) is permitted entry.
            config = get_config()
            if user.get("username") == config.SUPER_ADMIN_USERNAME:
                user["role"] = "super_admin"
                user["approved"] = True
                return user
            
            return {"error": "Access Denied: High Command clearance required."}
        return None

    async def list_pending_admins(self):
        if not self._redis_store: return []
        return await self._redis_store.list_pending_approvals()

    async def approve_admin(self, username: str):
        if not self._redis_store: return False
        return await self._redis_store.approve_user(username)


# --- SINGLETON ---
_session_manager: Optional[PersistentSessionManager] = None


async def get_session_manager() -> PersistentSessionManager:
    """Get or create session manager singleton."""
    global _session_manager
    
    if _session_manager is None:
        _session_manager = PersistentSessionManager()
    
    return _session_manager


async def init_session_manager(llm=None) -> PersistentSessionManager:
    """Initialize session manager at startup."""
    manager = await get_session_manager()
    await manager.start(llm=llm)
    return manager
