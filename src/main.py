"""
Sarah - Real Estate Voice Agent (Main Entry Point)

A voice-enabled AI agent for Madrid real estate using FastRTC, LangGraph, and Superlinked.

Production Features:
- Async-first design for non-blocking operations
- Comprehensive error handling with user-friendly feedback
- Unique session IDs per user
- Graceful degradation when services fail
- Structured logging throughout
"""

import sys
import asyncio
import uuid
import re
import warnings
import argparse
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Tuple, Optional

import numpy as np
import gradio as gr
from dotenv import load_dotenv
from fastrtc import Stream, ReplyOnPause
from langgraph.checkpoint.memory import InMemorySaver

# Ensure the root directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import config
from src.utils.logger import setup_logging, get_logger_with_context
from src.utils.exceptions import (
    ConfigurationError,
    ModelLoadError,
    TranscriptionError,
    AgentError,
    ToolExecutionError,
)
from src.tools.property_search import search_properties, initialize_search_engine
from src.models import load_stt_model, load_tts_model, get_llm, create_sarah_agent
from src.telephony import create_telephony_app


# --- INITIALIZATION ---
logger = setup_logging("Sarah-Main")

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")


class SarahAgent:
    """
    Encapsulates the Sarah voice agent with all its components.
    
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
    
    def __init__(self):
        self.stt_model = None
        self.tts_model = None
        self.agent = None
        self.is_ready = False
        self._session_checkpointer = InMemorySaver()
        
    def initialize(self) -> bool:
        """
        Initialize all components. Returns True if successful.
        
        This method is designed to fail fast during startup so issues
        are caught before users interact with the system.
        """
        logger.info("=" * 50)
        logger.info("🏠 SARAH VOICE AGENT - Starting Up")
        logger.info("=" * 50)
        
        # 1. Load environment and validate config
        load_dotenv()
        try:
            config.validate()
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
        
        # 4. Initialize search engine
        logger.info("🔍 Pre-loading search engine...")
        if not initialize_search_engine():
            logger.warning("⚠️ Search engine initialization failed - will retry on first search")
        
        # 5. Set up LLM and agent
        try:
            llm = get_llm()
            self.agent = create_sarah_agent(
                llm, 
                tools=[search_properties], 
                checkpointer=self._session_checkpointer
            )
        except Exception as e:
            # Logger handled in model modules
            return False
        
        self.is_ready = True
        logger.info("=" * 50)
        logger.info("✅ SARAH IS READY!")
        logger.info("=" * 50)
        return True
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID for each conversation."""
        return f"session_{uuid.uuid4().hex[:8]}"
    
    async def _transcribe_audio(self, audio: Tuple[int, np.ndarray]) -> Optional[str]:
        """
        Transcribe audio to text using the STT model.
        
        Runs in a thread pool to avoid blocking the event loop.
        """
        try:
            transcription = await asyncio.to_thread(self.stt_model.stt, audio)
            return transcription if transcription and transcription.strip() else None
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise TranscriptionError(e)
    
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
            # We use a loop here because the agent might want to use a tool,
            # then think again before giving a final answer.
            messages = [("user", user_text)]
            max_iterations = 5  # Prevent infinite loops
            
            for i in range(max_iterations):
                # 1. Ask the agent for a response
                result = await self.agent.ainvoke(
                    {"messages": messages},
                    {"configurable": {"thread_id": session_id}}
                )
                
                last_msg = result["messages"][-1]
                messages.append(last_msg)
                
                # 2. Check if agent wants to use a tool
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_call = last_msg.tool_calls[0]
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    request_logger.info(f"🔧 Iteration {i+1}: Agent calling tool {tool_name}")
                    
                    # Execute the tool
                    if tool_name == 'search_properties':
                        query = tool_args.get("user_request", "")
                        tool_result = await search_properties(query)
                    else:
                        tool_result = f"Unknown tool: {tool_name}"
                        request_logger.warning(f"Unknown tool requested: {tool_name}")
                    
                    # Add the tool result to messages and loop back to the agent
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call['id']
                    })
                    continue  # Tool used, ask the agent to think again
                
                # 3. If no tool call, this is the final answer
                return last_msg.content
                
            return "I'm having trouble finishing my thought. Could you try asking simpler?"
                
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            raise AgentError(str(e), e)
    
    def _clean_response_for_speech(self, text: str) -> str:
        """
        Clean the response text for TTS output.
        
        Removes any technical artifacts that might have slipped through.
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
    
    async def handle_voice_input(
        self, 
        audio: Tuple[int, np.ndarray]
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Main voice handler - processes audio input and generates voice response.
        
        This is the entry point called by FastRTC for each voice interaction.
        """
        # Generate unique session ID for this conversation
        session_id = self._generate_session_id()
        request_logger = get_logger_with_context("Sarah-Handler", request_id=session_id)
        
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
                return
            
            request_logger.info(f"🗣️ User: {user_text}")
            
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
                request_logger.info(f"🎙️ Sarah: {clean_response}")
                async for chunk in self._speak_text(clean_response):
                    yield chunk
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
def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Sarah Voice Agent")
    parser.add_argument("--phone", action="store_true", help="Start the Twilio telephony server")
    parser.add_argument("--call", type=str, help="Make an outbound call to this number (Format: +123456789)")
    args = parser.parse_args()
    
    # Create and initialize the agent
    sarah = SarahAgent()
    
    if not sarah.initialize():
        logger.critical("❌ Failed to initialize Sarah. Exiting.")
        sys.exit(1)
    
    if args.call:
        # 1. Start the Telephony Server in the background
        import uvicorn
        from src.config import get_config
        from src.telephony import create_telephony_app, make_outbound_call
        
        app_config = get_config()
        app = create_telephony_app(sarah)
        
        # Function to run the server
        def run_server():
            logger.info(f"📞 Starting Background Phone Server on {app_config.HOST}:{app_config.PORT}")
            uvicorn.run(app, host=app_config.HOST, port=app_config.PORT, log_level="error")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 2. Wait a moment for server to warm up
        logger.info("⏳ Waiting for server to stabilize...")
        time.sleep(3)
        
        # 3. Trigger the outbound call
        success = make_outbound_call(args.call)
        if success:
            print(f"✅ Sarah is now dialing {args.call}...")
            print("🎙️ Keep this window open while you talk!")
            # Keep the main thread alive while the call happens
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping Sarah...")
        else:
            print(f"❌ Failed to initiate call to {args.call}. Check logs/sarah.log for details.")
        sys.exit(0)
        
    if args.phone:
        # Start the Telephony Server (Twilio)
        import uvicorn
        from src.config import get_config
        app_config = get_config()
        
        app = create_telephony_app(sarah)
        logger.info(f"📞 Starting Phone Server on {app_config.HOST}:{app_config.PORT}")
        logger.info(f"🔗 Make sure Twilio points to: {app_config.SERVER_URL}/voice")
        uvicorn.run(app, host=app_config.HOST, port=app_config.PORT)
    else:
        # Start the FastRTC Web UI (Default)
        from fastrtc import Stream, ReplyOnPause
        stream = Stream(
            handler=ReplyOnPause(sarah.handle_voice_input),
            modality="audio",
            mode="send-receive"
        )
        
        logger.info("🌐 Launching web interface...")
        logger.info("💡 Open the URL shown below to talk to Sarah!")
        stream.ui.launch()


if __name__ == "__main__":
    # Option 1: Run with aiomonitor for async debugging (uncomment to use)
    # import aiomonitor
    # async def main_with_monitor():
    #     async with aiomonitor.start_monitor():
    #         main()
    # asyncio.run(main_with_monitor())
    
    # Option 2: Normal run (default)
    main()