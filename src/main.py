"""
Arya - Real Estate Voice Agent (Main Entry Point)

A voice-enabled AI agent for Madrid real estate using FastRTC, LangGraph, and Superlinked.

Production Features:
- Async-first design for non-blocking operations
- Comprehensive error handling with user-friendly feedback
- Unique session IDs per user
- Graceful degradation when services fail
- Structured logging throughout
"""

import os
import sys

# Windows Unicode Fix: Force UTF-8 for project config files ONLY
# We must NOT override open() globally as it breaks third-party packages like kokoro_onnx
os.environ["PYTHONUTF8"] = "1"

import asyncio
import uuid
import re
import warnings
import argparse
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Tuple, Optional

# --- PRODUCTION STABILITY PATCHES ---

# 1. Fix Python 3.12 / Anyio / Uvicorn conflict (loop_factory error)
if sys.version_info >= (3, 12):
    try:
        import uvicorn.config
        # Force uvicorn to ignore loop_factory which crashes in patched anyio environments
        uvicorn.config.Config.get_loop_factory = lambda self: None
        
        # Also try to un-patch asyncio.run if anyio has already touched it
        import anyio._backends._asyncio
        if hasattr(anyio._backends._asyncio, "run"):
             asyncio.run = anyio._backends._asyncio._original_run
    except Exception:
        # Fallback to a broader patch if specific imports fail
        _original_run = asyncio.run
        def _patched_run(main, *, debug=None, loop_factory=None):
            return _original_run(main, debug=debug)
        asyncio.run = _patched_run

# 2. Silence Pydantic V2/V1 mixing and YAML config warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*YamlConfigSettingsSource.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Mixing V1 models and V2 models.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import numpy as np
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

# Ensure the root directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import config, get_config
from src.utils.logger import setup_logging, get_logger_with_context
from src.utils.exceptions import (
    ConfigurationError,
    ModelLoadError,
    TranscriptionError,
    AgentError,
    ToolExecutionError,
)
from src.utils.retry import async_retry
# Use ultra-lightweight Numpy-based search
from src.tools.property_search_light import (
    search_properties, 
    initialize_lightweight_search
)
from src.models import load_stt_model, load_tts_model, get_llm, create_arya_agent

from src.telephony import create_telephony_app

# Memory system for persistent user recognition
from src.memory.session_manager import PersistentSessionManager
from src.memory.models import SessionMemory, IntentType


# --- INITIALIZATION ---
logger = setup_logging("Arya-Main")

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")


class AryaAgent:
    """
    Encapsulates the Arya voice agent with all its components.
    
    Benefits of using a class:
    - Clean lifecycle management
    - Easy testing (can create isolated instances)
    - No global mutable state
    - Clear dependency injection
    """
    
    # Error messages to speak when things go wrong
    ERROR_MESSAGES = {
        "transcription": "I couldn't quite catch that. Could you please repeat?",
        "agent": "I lost my train of thought for a moment. What were you saying?",
        "tool": "I had some trouble looking that up. Let me try again.",
        "general": "I ran into a small hiccup. Could you repeat that?",
        "startup": "I'm still warming up. Please give me a moment.",
    }
    
    def __init__(self, use_redis: bool = True):
        self.stt_model = None
        self.tts_model = None
        self.llm = None
        self.agent = None
        self.is_ready = False
        self._session_checkpointer = InMemorySaver()
        
        self.use_redis = use_redis
        self.session_manager: Optional[PersistentSessionManager] = None
        self._active_sessions: dict = {}  # session_id -> user_context
        self._last_responses: dict = {}   # session_id -> last_response_text
        self._last_user_texts: dict = {}  # session_id -> last_user_text
        
    def initialize(self) -> bool:
        """
        Initialize all components. Returns True if successful.
        
        This method is designed to fail fast during startup so issues
        are caught before users interact with the system.
        """
        logger.info("=" * 50)
        logger.info("🏠 ARYA VOICE AGENT - Starting Up")
        logger.info("=" * 50)
        
        # 1. Load environment and validate config (this also auto-detects ngrok)
        load_dotenv()
        try:
            app_config = get_config()
            app_config.validate()
        except Exception as e:
            logger.critical(f"❌ Configuration error: {e}")
            return False
        
        # 2. Load STT model
        try:
            self.stt_model = load_stt_model()
        except Exception as e:
            # Logger already handled in load_stt_model
            return False
        
        # 3. Load TTS model
        try:
            self.tts_model = load_tts_model()
        except Exception as e:
            # Logger already handled in load_tts_model
            return False
        
        # 4. Initialize search engine (Numpy - will be fully loaded in initialize_async)
        logger.info("🏢 Using ultra-lightweight Numpy search engine")
        
        # 5. Set up LLM and agent (with Chroma search tool)
        try:
            self.llm = get_llm()
            self.agent = create_arya_agent(
                self.llm, 
                tools=[search_properties], 
                checkpointer=self._session_checkpointer
            )
        except Exception as e:
            # Logger handled in model modules
            return False
        
        self.is_ready = True
        logger.info("=" * 50)
        logger.info("✅ ARYA IS READY!")
        logger.info("=" * 50)
        
        import gc
        gc.collect()  # Reclaim memory after heavy imports and setup
        logger.info("🧹 Garbage collection complete (Clean Start)")
        
        return True
    
    async def initialize_async(self) -> bool:
        """
        Initialize async components (Redis, session manager).
        Call this after initialize() in an async context.
        """
        if not self.is_ready:
            logger.warning("Cannot init async - sync init not complete")
            return False
        
        # 1. Initialize Numpy Search (loads data and computes embeddings)
        logger.info("🚀 Initializing Numpy search and embeddings...")
        if not await initialize_lightweight_search():
            logger.error("❌ Numpy search initialization failed")
            # We continue anyway as it might retry on first search, 
            # but this is a critical warning.
        
        # 2. Initialize Redis/Memory
        if self.use_redis:
            try:
                logger.info("🔗 Initializing persistent memory (Redis)...")
                self.session_manager = PersistentSessionManager(
                    use_redis=True,
                    auto_save_interval=30,
                    summarize_on_end=True
                )
                await self.session_manager.start(llm=self.llm)
                logger.info("✅ Redis memory connected!")
            except Exception as e:
                logger.warning(f"⚠️ Redis init failed: {e} - using in-memory only")
                self.session_manager = None
        
        return True
    
    async def start_session(
        self, 
        session_id: str, 
        phone_number: Optional[str] = None
    ) -> dict:
        """
        Start a new conversation session with user recognition.
        
        Returns context dict with user info and personalized greeting.
        """
        context = {
            "session_id": session_id,
            "is_returning": False,
            "greeting": None,
            "user_preferences": None,
        }
        
        if self.session_manager:
            try:
                user_context = await self.session_manager.start_session(
                    session_id=session_id,
                    phone_number=phone_number
                )
                context["is_returning"] = user_context.is_returning
                context["greeting"] = self.session_manager.get_user_greeting(user_context)
                context["user_preferences"] = user_context.preferences
                self._active_sessions[session_id] = user_context
                
                if user_context.is_returning:
                    logger.info(f"🔁 Returning user recognized: {user_context.user_id[:8]}...")
                else:
                    logger.info(f"🆕 New user: {user_context.user_id[:8]}...")
            except Exception as e:
                logger.error(f"Session start error: {e}")
        
        return context
    
    async def end_session(self, session_id: str) -> bool:
        """End session and persist to Redis."""
        if self.session_manager:
            try:
                success = await self.session_manager.end_session(session_id)
                if session_id in self._active_sessions:
                    del self._active_sessions[session_id]
                return success
            except Exception as e:
                logger.error(f"Session end error: {e}")
        return False
    
    def track_turn(self, session_id: str, role: str, content: str) -> bool:
        """Track a conversation turn for memory."""
        if self.session_manager:
            return self.session_manager.add_turn(session_id, role, content)
        return False
    
    def get_last_response(self, session_id: str) -> str:
        """Get the last response text for a specific session."""
        return self._last_responses.get(session_id, "")
    
    def get_last_user_text(self, session_id: str) -> str:
        """Get the last user transcription for a specific session."""
        return self._last_user_texts.get(session_id, "")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID for each conversation."""
        return f"session_{uuid.uuid4().hex[:8]}"
    
    async def _transcribe_audio(self, audio: Tuple[int, np.ndarray]) -> Optional[str]:
        """
        Transcribe audio to text using the STT model.
        Supports both sync and async models.
        """
        try:
            # Check if stt is an async method
            if asyncio.iscoroutinefunction(self.stt_model.stt):
                transcription = await self.stt_model.stt(audio)
            else:
                # Fallback to thread pool for sync models
                transcription = await asyncio.to_thread(self.stt_model.stt, audio)
            
            return transcription if transcription and transcription.strip() else None
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise TranscriptionError(e)
    
    @async_retry(tries=3, delay=0.5, backoff=2.0)
    async def _get_agent_response(
        self, 
        user_text: str, 
        session_id: str,
        request_logger
    ) -> str:
        """
        Get a response from the agent, handling tool calls automatically in a loop.
        """
        try:
            # 1. Fetch persistent context for this user (preferences, summary, last turns)
            full_context = ""
            if self.session_manager:
                full_context = self.session_manager.get_context_for_llm(session_id)
            
            # 2. Build the messages with context injection
            messages = []
            if full_context:
                # Add context as a system-like separator to remind the LLM of who it's talking to
                messages.append(("user", f"[SYSTEM CONTEXT: {full_context}]\n\nUser's message: {user_text}"))
            else:
                messages.append(("user", user_text))

            # 3. Ask the agent for a response
            result = await self.agent.ainvoke(
                {"messages": messages},
                {"configurable": {"thread_id": session_id}}
            )
            
            # The last message is the final AI response
            return result["messages"][-1].content
                
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            raise AgentError(str(e), e)
    
    def _number_to_words(self, num: int) -> str:
        """Convert an integer to spoken words (English)."""
        if num == 0:
            return "zero"
        
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", 
                "eight", "nine", "ten", "eleven", "twelve", "thirteen", 
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", 
                "sixty", "seventy", "eighty", "ninety"]
        
        def below_thousand(n):
            if n < 20:
                return ones[n]
            elif n < 100:
                return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
            else:
                return ones[n // 100] + " hundred" + (" " + below_thousand(n % 100) if n % 100 else "")
        
        if num >= 1_000_000_000:
            return f"{below_thousand(num // 1_000_000_000)} billion" + \
                   (" " + self._number_to_words(num % 1_000_000_000) if num % 1_000_000_000 else "")
        elif num >= 1_000_000:
            return f"{below_thousand(num // 1_000_000)} million" + \
                   (" " + self._number_to_words(num % 1_000_000) if num % 1_000_000 else "")
        elif num >= 1000:
            return f"{below_thousand(num // 1000)} thousand" + \
                   (" " + self._number_to_words(num % 1000) if num % 1000 else "")
        else:
            return below_thousand(num)
    
    def _clean_response_for_speech(self, text: str) -> str:
        """
        Clean the response text for TTS output.
        
        - Removes technical artifacts (JSON, markdown, function tags)
        - Converts numbers to spoken words for natural pronunciation
        - Handles currency symbols and common abbreviations
        """
        if not text:
            return ""
        
        # Remove function tags
        text = re.sub(r'<function.*?>.*?</function>', '', text, flags=re.DOTALL)
        # Remove JSON objects
        text = re.sub(r'\{[^}]+\}', '', text)
        # Remove markdown artifacts
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code blocks
        
        # Convert currency with numbers: €500,000 -> five hundred thousand euros
        def currency_to_words(match):
            symbol = match.group(1)
            number_str = match.group(2).replace(',', '').replace('.', '')
            try:
                number = int(number_str)
                words = self._number_to_words(number)
                currency = "euros" if symbol == "€" else "dollars"
                return f"{words} {currency}"
            except ValueError:
                return match.group(0)
        
        text = re.sub(r'(€|\$)\s*([\d,\.]+)', currency_to_words, text)
        
        # Convert standalone large numbers (likely prices)
        def number_to_words_match(match):
            number_str = match.group(0).replace(',', '')
            try:
                number = int(number_str)
                # Only convert numbers >= 1000 (likely prices/stats)
                if number >= 1000:
                    return self._number_to_words(number)
                return match.group(0)
            except ValueError:
                return match.group(0)
        
        # Match numbers with optional commas (e.g., 500,000 or 500000)
        text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b', number_to_words_match, text)
        
        # Handle common abbreviations
        text = text.replace(" sqm", " square meters")
        text = text.replace(" sq.m.", " square meters")
        text = text.replace(" m²", " square meters")
        text = text.replace(" sq ft", " square feet")
        text = text.replace(" bd", " bedroom")
        text = text.replace(" br", " bedroom")
        text = text.replace("apt", "apartment")
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    async def _speak_text(
        self, 
        text: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Convert text to speech and yield audio chunks.
        """
        try:
            async for audio_chunk in self.tts_model.stream_tts(text):
                yield audio_chunk
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            # Can't speak the error if TTS is broken, so just log it
    
    async def handle_text_input(
        self, 
        user_text: str,
        session_id: Optional[str] = None,
        stop_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Processes text input directly (skipping STT) and generates voice response.
        Useful for Streaming STT where transcription happens concurrently.
        """
        session_id = session_id or self._generate_session_id()
        request_logger = get_logger_with_context("Arya-TextHandler", request_id=session_id)
        
        if not self.is_ready:
            async for chunk in self._speak_text(self.ERROR_MESSAGES["startup"]):
                yield chunk
            return
            
        if not user_text:
            return

        try:
            request_logger.info(f"🗣️ User (via Stream): {user_text}")
            self._last_user_texts[session_id] = user_text
            
            # Track user turn for memory
            self.track_turn(session_id, "user", user_text)
            
            # Step 2: Get agent response
            request_logger.info("🤔 Thinking...")
            
            # Start timer for pacing filler
            response_task = asyncio.create_task(self._get_agent_response(
                user_text, 
                session_id, 
                request_logger
            ))
            
            # Wait with timeout to send a filler if it takes too long
            try:
                ai_response = await asyncio.wait_for(asyncio.shield(response_task), timeout=1.8)
            except asyncio.TimeoutError:
                # Send a filler to the user
                import random
                fillers = ["Hmm, let me check that for you...", "Interesting, let me see...", "One moment, searching...", "Hmm..."]
                filler_text = random.choice(fillers)
                request_logger.info(f"⏳ Sending pacing filler: {filler_text}")
                async for chunk in self._speak_text(filler_text):
                    yield chunk
                
                # Now wait for the real response
                ai_response = await response_task
            
            # Step 3: Clean and speak response
            clean_response = self._clean_response_for_speech(ai_response)
            
            if clean_response:
                request_logger.info(f"🎙️ Arya: {clean_response}")
                self._last_responses[session_id] = clean_response
                
                full_response_delivered = True
                try:
                    async for chunk in self._speak_text(clean_response):
                        if stop_event and stop_event.is_set():
                            request_logger.info("⚡ Audio streaming interrupted by stop_event")
                            full_response_delivered = False
                            break
                        yield chunk
                except Exception as e:
                    request_logger.error(f"Error during audio streaming: {e}")
                    full_response_delivered = False
                
                if full_response_delivered:
                    self.track_turn(session_id, "assistant", clean_response)
                else:
                    self.track_turn(session_id, "assistant", "[Interrupted before completion]")
                    try:
                        await self.agent.aupdate_state(
                            {"configurable": {"thread_id": session_id}},
                            {"messages": [AIMessage(content="[System Note: Arya was interrupted here.]")]}
                        )
                    except: pass
            else:
                request_logger.warning("Empty response from agent")
                
        except AgentError as e:
            request_logger.error(f"Agent error: {e}")
            async for chunk in self._speak_text(self.ERROR_MESSAGES["agent"]):
                yield chunk
        except Exception as e:
            request_logger.error(f"Unexpected error: {e}", exc_info=True)
            async for chunk in self._speak_text(self.ERROR_MESSAGES["general"]):
                yield chunk

    async def handle_voice_input(
        self, 
        audio: Tuple[int, np.ndarray],
        session_id: Optional[str] = None,
        stop_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Main voice handler - processes audio input and generates voice response.
        
        Args:
            audio: Tuple of (sample_rate, audio_data)
            session_id: Unique ID for the conversation
            stop_event: Optional event to signal interruption
        """
        # Use provided session_id (e.g., Twilio Stream SID) or generate a new one
        session_id = session_id or self._generate_session_id()
        request_logger = get_logger_with_context("Arya-Handler", request_id=session_id)
        
        if not self.is_ready:
            request_logger.warning("Agent not ready, speaking startup message")
            async for chunk in self._speak_text(self.ERROR_MESSAGES["startup"]):
                yield chunk
            return
        
        try:
            # Step 1: Transcribe audio
            request_logger.info("🎤 Transcribing...")
            user_text = await self._transcribe_audio(audio)
            
            if not user_text:
                request_logger.debug("Empty transcription, ignoring")
                # Clear last response if current input is empty/noise
                if session_id in self._last_responses:
                    self._last_responses[session_id] = None
                return
            
            request_logger.info(f"🗣️ User: {user_text}")
            self._last_user_texts[session_id] = user_text
            
            # Track user turn for memory
            self.track_turn(session_id, "user", user_text)
            
            # Step 2: Get agent response
            request_logger.info("🤔 Thinking...")
            ai_response = await self._get_agent_response(
                user_text, 
                session_id, 
                request_logger
            )
            
            # Step 3: Clean and speak response
            clean_response = self._clean_response_for_speech(ai_response)
            
            if clean_response:
                request_logger.info(f"🎙️ Arya: {clean_response}")
                self._last_responses[session_id] = clean_response
                
                # Track assistant turn ONLY if we finish speaking
                # (Postponed from here to prevent context leakage during barge-in)
                
                full_response_delivered = True
                try:
                    async for chunk in self._speak_text(clean_response):
                        if stop_event and stop_event.is_set():
                            request_logger.info("⚡ Audio streaming interrupted by stop_event")
                            full_response_delivered = False
                            break
                        yield chunk
                except Exception as e:
                    request_logger.error(f"Error during audio streaming: {e}")
                    full_response_delivered = False
                
                if full_response_delivered:
                    # Successfully finished speaking - add to memory
                    self.track_turn(session_id, "assistant", clean_response)
                else:
                    # Interrupted - record as partial if possible, or just skip
                    request_logger.info("🚫 Turn interrupted, not adding full response to persistent memory")
                    self.track_turn(session_id, "assistant", "[Interrupted before completion]")
                    
                    # Also update the internal LangGraph memory so Sarah knows she was cut off
                    # This prevents her from referring to things the user didn't hear
                    try:
                        await self.agent.aupdate_state(
                            {"configurable": {"thread_id": session_id}},
                            {"messages": [AIMessage(content="[System Note: Arya was interrupted by the user here. The previous message was cut off and the user might not have heard everything.]")]}
                        )
                        request_logger.debug("✅ LangGraph memory updated with interruption note")
                    except Exception as e:
                        request_logger.warning(f"⚠️ Failed to update LangGraph memory: {e}")
            else:
                request_logger.warning("Empty response from agent")
                
        except TranscriptionError:
            request_logger.warning("Transcription failed, speaking error message")
            async for chunk in self._speak_text(self.ERROR_MESSAGES["transcription"]):
                yield chunk
                
        except AgentError as e:
            request_logger.error(f"Agent error: {e}")
            async for chunk in self._speak_text(self.ERROR_MESSAGES["agent"]):
                yield chunk
                
        except Exception as e:
            request_logger.error(f"Unexpected error: {e}", exc_info=True)
            async for chunk in self._speak_text(self.ERROR_MESSAGES["general"]):
                yield chunk


# --- APPLICATION ENTRY POINT ---
async def main_async():
    """Async wrapper for the application."""
    parser = argparse.ArgumentParser(description="Arya Voice Agent")
    parser.add_argument("--phone", action="store_true", help="Start the Twilio telephony server")
    parser.add_argument("--call", type=str, help="Make an outbound call to this number (Format: +123456789)")
    args = parser.parse_args()
    
    # Create and initialize the agent
    arya = AryaAgent()
    
    if not arya.initialize():
        logger.critical("❌ Failed to initialize Arya. Exiting.")
        sys.exit(1)
        
    # We do NOT call sarah.initialize_async() here because 
    # the telephony app's lifespan handler will do it for us.
    
    if args.call:
        # 1. Start the Telephony Server in the background
        import uvicorn
        import socket
        from src.config import get_config
        from src.telephony import create_telephony_app, make_outbound_call
        
        app_config = get_config()

        from src.telephony import create_telephony_app, make_outbound_call
        app_config = get_config()

        app = create_telephony_app(arya)
        
        config_uv = uvicorn.Config(app, host=app_config.HOST, port=app_config.PORT, log_level="error")
        server = uvicorn.Server(config_uv)
        
        logger.info(f"📞 Starting Background Phone Server on {app_config.HOST}:{app_config.PORT}")
        # Run in a separate task so we can proceed to trigger calls
        server_task = asyncio.create_task(server.serve())
        
        logger.info("⏳ Waiting for server to stabilize...")
        await asyncio.sleep(3)
        
        # 3. Trigger multiple outbound calls
        phone_numbers = [n.strip() for n in args.call.split(",")]
        
        for number in phone_numbers:
            logger.info(f"🚀 Initiating call to {number}...")
            # make_outbound_call is likely synchronous Twilio SDK call
            success = await make_outbound_call(number)
            if success:
                print(f"✅ Arya is now dialing {number}...")
            else:
                print(f"❌ Failed to initiate call to {number}.")
            await asyncio.sleep(1)

        print("🎙️ Keep this window open while the calls are active!")
        try:
            await server_task
        except asyncio.CancelledError:
            print("\n🛑 Stopping Arya...")
        
    elif args.phone:
        # Start the Telephony Server (Twilio)
        from src.config import get_config
        app_config = get_config()
        from src.telephony import create_telephony_app
        
        # Proactive: Clear port if something is hanging from a previous run
        def clear_port(port):
            import subprocess
            import os
            import signal
            logger.info(f"🔍 Checking for lingering processes on port {port}...")
            try:
                # Find the PID using netstat
                output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
                for line in output.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        pid = line.strip().split()[-1]
                        if pid != "0":
                            logger.warning(f"⚠️ Killing process {pid} occupying port {port}...")
                            if os.name == 'nt':
                                subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                            else:
                                os.kill(int(pid), signal.SIGKILL)
                            import time
                            time.sleep(1) # Give OS time to release resources
            except subprocess.CalledProcessError:
                pass # Port is clean
            except Exception as e:
                logger.error(f"Error clearing port: {e}")

        clear_port(app_config.PORT)
        
        app = create_telephony_app(arya)
        
        # Determine loggable URL (convert 0.0.0.0 to clickable localhost)
        display_host = "127.0.0.1" if app_config.HOST == "0.0.0.0" else app_config.HOST
        logger.info(f"📞 Starting Phone Server on {app_config.HOST}:{app_config.PORT}")
        logger.info(f"🖥️  Admin Dashboard: http://{display_host}:{app_config.PORT}/admin")
        logger.info(f"🔗 Twilio Webhook URL: {app_config.SERVER_URL}/voice")
        
        config_uv = uvicorn.Config(app, host=app_config.HOST, port=app_config.PORT)

        server = uvicorn.Server(config_uv)
        await server.serve()
    else:
        # Start the FastRTC Web UI (Default)
        try:
            from fastrtc import Stream, ReplyOnPause
            import gradio as gr
        except ImportError:
            logger.error("❌ FastRTC/Gradio not installed. Use --phone mode for Twilio calls.")
            logger.error("   To enable web mode, run: pip install fastrtc gradio librosa numba")
            return
            
        stream = Stream(
            handler=ReplyOnPause(arya.handle_voice_input),
            modality="audio",
            mode="send-receive"
        )
        
        logger.info("🌐 Launching web interface...")
        logger.info("💡 Open the URL shown below to talk to Arya!")
        
        # Gradio launch is blocking, we run it in a thread or just call it
        # Since this is the end of the entry point, we can just call it
        await asyncio.to_thread(stream.ui.launch)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
