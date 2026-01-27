"""
Telephony Module for Sarah Voice Agent.

Handles Twilio Media Streams, Webhooks, and Outbound Calls.

Enhanced Features:
- Concurrent processing for low-latency interruptions
- Adaptive silence detection (600ms baseline, adaptive to speaker)
- RMS monitoring for barge-in support
- Soft full-duplex for natural interruptions
"""

import json
import base64
import asyncio
import numpy as np
import audioop
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect

from src.config import get_config
from src.utils.logger import setup_logging
from src.audio.rms_monitor import RMSMonitor
from src.audio.adaptive_vad import AdaptiveSilenceDetector
from src.audio.barge_in import BargeInHandler, ConversationState
from src.audio.webrtc_vad import SmartBargeInDetector, WEBRTC_AVAILABLE, VADAggressiveness
from src.dashboard_template import DASHBOARD_HTML
from src.dashboard_utils import get_recent_logs, get_current_config_map, update_env_file

logger = setup_logging("Telephony")
config = get_config()

# Twilio Client for Outbound Calls
twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN) if config.TWILIO_ACCOUNT_SID else None


def create_telephony_app(agent):
    """
    Creates a FastAPI app to handle Twilio calls with enhanced audio processing.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: initialize agent async components
        if hasattr(agent, 'initialize_async'):
            logger.info("🚀 Calling agent.initialize_async()...")
            await agent.initialize_async()
        yield
        # Shutdown: cleanup
        if hasattr(agent, 'session_manager') and agent.session_manager:
            await agent.session_manager.stop()

    app = FastAPI(lifespan=lifespan)

    # Rate limiting & Capacity
    MAX_CONCURRENT_CALLS = 5
    _active_call_count = 0
    _active_lock = asyncio.Lock()

    # Store caller info for WebSocket sessions
    _pending_calls: dict = {}  # call_sid -> caller_info
    _active_sessions: dict = {} # stream_sid -> session_details (for dashboard)

    async def _cleanup_pending_calls():
        """Periodically remove call data that never established a WebSocket."""
        while True:
            try:
                await asyncio.sleep(600)  # Clean every 10 mins
                now = asyncio.get_event_loop().time()
                to_delete = [
                    sid for sid, info in _pending_calls.items() 
                    if now - info["timestamp"] > 3600  # 1 hour TTL
                ]
                for sid in to_delete:
                    del _pending_calls[sid]
                    logger.debug(f"🧹 Cleaned up expired call data: {sid[:8]}")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(_cleanup_pending_calls())

    @app.get("/health")
    async def health_check_root():
        """Basic health check for monitoring and deployment."""
        return {
            "status": "healthy",
            "active_calls": _active_call_count,
            "max_capacity": MAX_CONCURRENT_CALLS,
            "agent_ready": agent.is_ready if hasattr(agent, 'is_ready') else False
        }

    # --- DASHBOARD ROUTES ---

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard():
        """Serve the Sarah Admin Dashboard."""
        return DASHBOARD_HTML

    @app.get("/dashboard/status")
    async def get_dashboard_status():
        """Return live status for the dashboard."""
        sessions_list = []
        for sid, data in _active_sessions.items():
            sessions_list.append({
                "sid": sid,
                "phone": data.get("phone", "Unknown"),
                "last_text": data.get("last_text", ""),
                "duration": f"{int(asyncio.get_event_loop().time() - data.get('start_time', 0))}s"
            })
            
        return {
            "active_calls": _active_call_count,
            "agent_ready": agent.is_ready if hasattr(agent, 'is_ready') else False,
            "sessions": sessions_list
        }

    @app.get("/dashboard/logs")
    async def get_dashboard_logs():
        """Return recent system logs."""
        return get_recent_logs(100)

    @app.get("/dashboard/config")
    async def get_dashboard_config():
        """Return current application configuration."""
        return get_current_config_map()

    @app.post("/dashboard/config")
    async def post_dashboard_config(updates: dict):
        """Update configuration in .env and notify of restart."""
        success = update_env_file(updates)
        return {"status": "success" if success else "failed"}

    @app.post("/voice")
    async def handle_incoming_call(request: Request):
        """
        Endpoint for Twilio's 'Voice URL'. 
        Tells Twilio to connect to our WebSocket for audio streaming.
        Also extracts caller phone number for user recognition.
        """
        # Extract form data from Twilio
        form_data = await request.form()
        caller_phone = form_data.get("From", "")  # Caller's phone number
        called_phone = form_data.get("To", "")
        call_sid = form_data.get("CallSid", "")
        
        logger.info(f"📞 Incoming call: {caller_phone} → {called_phone} (SID: {call_sid[:8]}...)")
        
        # Check capacity
        if _active_call_count >= MAX_CONCURRENT_CALLS:
            logger.warning(f"🚫 Rejecting call {call_sid[:8]}: System at capacity ({MAX_CONCURRENT_CALLS})")
            response = VoiceResponse()
            response.say("I'm sorry, Sarah is currently helping other customers. Please try again in a few minutes.")
            response.hangup()
            return Response(content=str(response), media_type="application/xml")
            
        # Store caller info for the WebSocket to retrieve
        _pending_calls[call_sid] = {
            "phone": caller_phone,
            "called": called_phone,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        response = VoiceResponse()
        # Say a quick greeting before passing to Sarah
        response.say("Connecting you to Sarah...")
        
        # --- Clean URL Handling ---
        # Ensure we don't have double slashes or missing subdomains
        base_url = config.SERVER_URL.strip().rstrip('/')
        websocket_domain = base_url.replace('https://', '').replace('http://', '')
        stream_url = f"wss://{websocket_domain}/media-stream"
        
        logger.info(f"🔗 Twilio Stream URL: {stream_url}")
        
        connect = Connect()
        connect.stream(url=stream_url)
        response.append(connect)
        
        return Response(content=str(response), media_type="application/xml")

    @app.websocket("/media-stream")
    async def handle_media_stream(websocket: WebSocket):
        """
        Refactored handler with concurrent hearing/speaking tasks.
        """
        nonlocal _active_call_count
        
        async with _active_lock:
            _active_call_count += 1
            
        try:
            await websocket.accept()
            logger.info(f"🎙️ WebSocket connected (Active Calls: {_active_call_count})")
            
            # --- Initialize Components ---
            # Lowered thresholds for better sensitivity to quiet voices
            # 8kHz μ-law from Twilio has limited dynamic range
            rms_monitor = RMSMonitor(
                silence_threshold=config.RMS_SILENCE_THRESHOLD,  # 500 default
                barge_in_threshold=config.RMS_BARGE_IN_THRESHOLD,  # 800 default
                window_size=20  # Shorter window for faster response
            )
            
            adaptive_vad = AdaptiveSilenceDetector(
                base_silence_ms=config.SILENCE_THRESHOLD_MS,
                min_silence_ms=config.SILENCE_THRESHOLD_MIN_MS,
                max_silence_ms=config.SILENCE_THRESHOLD_MAX_MS
            )
            
            # Use WebRTC VAD for smart barge-in (filters coughs/sneezes)
            smart_barge_in = SmartBargeInDetector(
                sample_rate=8000,  # Twilio sends 8kHz audio
                aggressiveness=VADAggressiveness.AGGRESSIVE, # Use enum for clarity
                rms_threshold=config.RMS_BARGE_IN_THRESHOLD,
                confirmation_frames=config.BARGE_IN_CONFIRM_FRAMES,
                cooldown_ms=500,
                grace_period_ms=getattr(config, 'BARGE_IN_GRACE_PERIOD_MS', 0)
            )
            
            # Legacy barge-in handler (fallback if WebRTC unavailable)
            barge_in_handler = BargeInHandler(
                rms_monitor=rms_monitor,
                barge_in_threshold=config.RMS_BARGE_IN_THRESHOLD,
                confirmation_frames=config.BARGE_IN_CONFIRM_FRAMES
            )
            
            # Log which VAD method is active
            if WEBRTC_AVAILABLE:
                logger.info("🧠 Using WebRTC VAD for intelligent barge-in detection")
            else:
                logger.info("📊 Using RMS-based barge-in detection (install webrtcvad for better accuracy)")
            
            # Audio gain for quiet voices (2x boost)
            AUDIO_GAIN = 2.0
            
            # --- State & Queues ---
            stream_sid = None
            call_sid = None
            caller_phone = None
            audio_queue = asyncio.Queue()
            
            # Shared control variables
            stop_playback = asyncio.Event()
            is_sarah_speaking = False
            session_started = False
            
            PACKET_DURATION_MS = 20
            MIN_AUDIO_MS = 200
            
            def check_if_question(text: str) -> bool:
                if not text: return False
                clean_text = text.strip()
                return clean_text.endswith('?') or any(clean_text.lower().startswith(q) for q in ['who', 'what', 'where', 'when', 'why', 'how', 'can', 'could', 'would', 'is', 'are', 'do', 'does'])

            async def send_to_twilio(audio_chunk_np: np.ndarray, source_sample_rate: int):
                """Convert and send audio to Twilio."""
                if not stream_sid: return
                try:
                    audio_int16 = (audio_chunk_np * 32767).astype(np.int16) if audio_chunk_np.dtype == np.float32 else audio_chunk_np.astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    resampled, _ = audioop.ratecv(audio_bytes, 2, 1, source_sample_rate, 8000, None)
                    mulaw_data = audioop.lin2ulaw(resampled, 2)
                    payload = base64.b64encode(mulaw_data).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload}
                    }))
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")

            async def start_user_session(sid):
                nonlocal caller_phone, session_started, is_sarah_speaking
                if sid in _pending_calls:
                    caller_phone = _pending_calls[sid].get("phone")
                    del _pending_calls[sid]
                
                context = await agent.start_session(session_id=stream_sid, phone_number=caller_phone)
                greeting = context.get("greeting") or (
                    "Welcome back! I'm Sarah, your real estate assistant. How can I help you today?" 
                    if context.get("is_returning") else 
                    "Hello! I'm Sarah, your real estate assistant in Madrid. How can I help you today?"
                )
                
                logger.info(f"🎙️ Speaking greeting: {greeting[:40]}...")
                is_sarah_speaking = True
                barge_in_handler.start_playback()
                smart_barge_in.start_monitoring()  # Enable WebRTC VAD monitoring
                try:
                    async for chunk in agent._speak_text(greeting):
                        if stop_playback.is_set(): break
                        await send_to_twilio(chunk[1], chunk[0])
                finally:
                    is_sarah_speaking = False
                    barge_in_handler.stop_playback()
                    smart_barge_in.stop_monitoring()  # Disable monitoring
                    session_started = True
                    # Store session start time for dashboard
                    _active_sessions[stream_sid] = {
                        "phone": caller_phone,
                        "start_time": asyncio.get_event_loop().time(),
                        "last_text": "Session Started"
                    }

            async def speaker_task():
                """Task that pulls from queue and speaks. Supports true barge-in."""
                nonlocal is_sarah_speaking
                while True:
                    try:
                        audio_data = await audio_queue.get()
                        stop_playback.clear()  # Reset stop flag for new turn
                        
                        logger.info(f"🗣️ Processing user phrase...")
                        if stream_sid in _active_sessions:
                            _active_sessions[stream_sid]["last_text"] = "Processing phrase..."
                        resampled_input, _ = audioop.ratecv(audio_data, 2, 1, 8000, 16000, None)
                        
                        is_sarah_speaking = True
                        barge_in_handler.start_playback()
                        smart_barge_in.start_monitoring()
                        
                        try:
                            # Update dashboard with latest user text if available
                            if stream_sid in _active_sessions:
                                user_text = agent.get_last_user_text(stream_sid)
                                if user_text:
                                    _active_sessions[stream_sid]["last_text"] = f"User: {user_text}"

                            async for chunk in agent.handle_voice_input(
                                (16000, np.frombuffer(resampled_input, dtype=np.int16)),
                                session_id=stream_sid,
                                stop_event=stop_playback
                            ):
                                # Update dashboard with AI response once it starts speaking
                                if stream_sid in _active_sessions:
                                    ai_text = agent.get_last_response(stream_sid)
                                    if ai_text:
                                        _active_sessions[stream_sid]["last_text"] = f"Sarah: {ai_text}"
                                # Check for barge-in BEFORE sending each chunk
                                if stop_playback.is_set():
                                    logger.info("⚡ Interruption detected in speaker task. Stopping.")
                                    is_sarah_speaking = False # Force false immediately
                                    # Clear Twilio's audio buffer
                                    try:
                                        await websocket.send_text(json.dumps({
                                            "event": "clear",
                                            "streamSid": stream_sid
                                        }))
                                    except Exception:
                                        pass
                                    break
                                await send_to_twilio(chunk[1], chunk[0])
                        finally:
                            is_sarah_speaking = False
                            barge_in_handler.stop_playback()
                            smart_barge_in.stop_monitoring()
                            audio_queue.task_done()
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error in speaker task: {e}", exc_info=True)

            # Start the speaker in the background
            speaker = asyncio.create_task(speaker_task())
            audio_buffer = bytearray()
            barge_in_audio_buffer = bytearray()  # Buffer for speech during barge-in
            is_in_barge_in_mode = False  # Track if we're collecting barge-in speech

            # Main Hearing Loop
            while True:
                message = await websocket.receive_text()
                packet = json.loads(message)
                
                if packet['event'] == 'start':
                    stream_sid = packet['start']['streamSid']
                    call_sid = packet['start'].get('callSid', '')
                    logger.info(f"🚀 Stream started: {stream_sid}")
                    if call_sid:
                        asyncio.create_task(start_user_session(call_sid))
                    
                elif packet['event'] == 'media':
                    pcm_audio = audioop.ulaw2lin(base64.b64decode(packet['media']['payload']), 2)
                    
                    # Apply gain boost for better detection of quiet voices
                    try:
                        pcm_audio = audioop.mul(pcm_audio, 2, AUDIO_GAIN)
                    except audioop.error:
                        pass
                    
                    rms_analysis = rms_monitor.process_audio(pcm_audio)
                    
                    # 1. BARGE-IN COLLECTION MODE: Continue collecting the user's new message
                    # This must have priority over is_sarah_speaking to handle transitions correctly
                    if is_in_barge_in_mode:
                        vad_result = adaptive_vad.update(
                            rms=rms_analysis.current_rms,
                            is_speech=rms_analysis.is_speaking,
                            packet_duration_ms=PACKET_DURATION_MS
                        )
                        
                        if rms_analysis.is_speaking:
                            barge_in_audio_buffer.extend(pcm_audio)
                        
                        # When user finishes their barge-in speech, queue it
                        if vad_result["pause_complete"] and len(barge_in_audio_buffer) > (MIN_AUDIO_MS * 16):
                            logger.info("✨ Barge-in speech complete, queuing NEW context.")
                            await audio_queue.put(bytes(barge_in_audio_buffer))
                            barge_in_audio_buffer = bytearray()
                            is_in_barge_in_mode = False
                            adaptive_vad.reset()
                    
                    # 2. BARGE-IN DETECTION: Only when Sarah is speaking
                    elif is_sarah_speaking:
                        is_barge_in, vad_info = smart_barge_in.check_barge_in(pcm_audio)
                        
                        if is_barge_in:
                            logger.info(f"🚨 BARGE-IN! Method={vad_info.method}, RMS={vad_info.rms_level:.0f}")
                            
                            # 1. Signal to stop playback
                            stop_playback.set()
                            is_sarah_speaking = False # Force false immediately for the next packet
                            
                            # 2. Send clear to Twilio immediately from here too
                            try:
                                await websocket.send_text(json.dumps({
                                    "event": "clear",
                                    "streamSid": stream_sid
                                }))
                            except Exception:
                                pass

                            # 3. Clear any pending items in the queue
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                    audio_queue.task_done()
                                except asyncio.QueueEmpty:
                                    break
                            
                            # 4. Enter barge-in collection mode
                            is_in_barge_in_mode = True
                            # Keep the audio we already collected leading up to this point
                            # barge_in_audio_buffer already contains the frames before confirmation
                            audio_buffer = bytearray()  # Clear normal buffer
                            adaptive_vad.reset()
                        
                        # While Sarah is speaking but no barge-in, still buffer audio
                        # This allows us to catch words spoken just before barge-in triggers
                        elif rms_analysis.is_speaking: # Changed to elif
                            barge_in_audio_buffer.extend(pcm_audio)
                    
                    # 3. NORMAL MODE: Sarah is silent, collect user speech
                    else:
                        is_question = check_if_question(agent.get_last_response(stream_sid))
                        vad_result = adaptive_vad.update(
                            rms=rms_analysis.current_rms,
                            is_speech=rms_analysis.is_speaking,
                            packet_duration_ms=PACKET_DURATION_MS
                        )
                        
                        if rms_analysis.is_speaking:
                            # If we have residual audio from a previous turn (e.g. user started talking just as Sarah finished)
                            if barge_in_audio_buffer:
                                audio_buffer.extend(barge_in_audio_buffer)
                                barge_in_audio_buffer = bytearray()
                            audio_buffer.extend(pcm_audio)
                        
                        if vad_result["pause_complete"] and len(audio_buffer) > (MIN_AUDIO_MS * 16):
                            logger.info("✨ User phrase captured, queuing for agent.")
                            await audio_queue.put(bytes(audio_buffer))
                            audio_buffer = bytearray()
                            adaptive_vad.reset()
                            if hasattr(agent, 'last_response_text'): agent.last_response_text = None
                        
                elif packet['event'] == 'stop':
                    logger.info("⏹️ Stream stopped")
                    speaker.cancel()
                    break
        except Exception as e:
            logger.error(f"❌ Media Stream error: {e}", exc_info=True)
        finally:
            # Properly decrement active call counter
            async with _active_lock:
                _active_call_count = max(0, _active_call_count - 1)
            
            if stream_sid and hasattr(agent, 'end_session'):
                await agent.end_session(stream_sid)
            if stream_sid in _active_sessions:
                del _active_sessions[stream_sid]
            logger.info(f"🔌 WebSocket closed (Active Calls: {_active_call_count})")

    # Note: /health endpoint is already defined above at line 84-92 with more details

    return app

async def make_outbound_call(to_number: str):
    if not twilio_client: return False
    try:
        loop = asyncio.get_event_loop()
        call = await loop.run_in_executor(
            None, 
            lambda: twilio_client.calls.create(
                from_=config.TWILIO_PHONE_NUMBER,
                to=to_number,
                url=f"{config.SERVER_URL}/voice"
            )
        )
        logger.info(f"🚀 Outbound call initiated: {call.sid}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to make outbound call: {e}")
        return False
