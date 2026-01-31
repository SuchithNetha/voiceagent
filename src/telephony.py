"""
Telephony Module for Arya Voice Agent.

Handles Twilio Media Streams, Webhooks, and Outbound Calls.

Enhanced Features:
- Concurrent processing for low-latency interruptions
- Adaptive silence detection (600ms baseline, adaptive to speaker)
- RMS monitoring for barge-in support
- Soft full-duplex for natural interruptions
"""

import os
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
from src.dashboard_template import PUBLIC_DASHBOARD, ADMIN_DASHBOARD, LOGIN_HTML
from src.utils.logger import setup_logging
from src.utils.auth import get_current_user, login_required, admin_required, create_access_token
from src.utils.email_sender import send_admin_email

logger = setup_logging("Telephony")
config = get_config()

# Twilio Client for Outbound Calls and Webhook Sync
twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN) if config.TWILIO_ACCOUNT_SID else None

async def sync_twilio_webhook():
    """
    Automatically synchronize the Twilio phone number's Voice URL 
    to point to the current server's public URL (ngrok/Render).
    This eliminates the need to manually update the Twilio Console.
    """
    if not twilio_client or not config.TWILIO_PHONE_NUMBER:
        return
    
    try:
        # Find the incoming phone number SID
        loop = asyncio.get_event_loop()
        numbers = await loop.run_in_executor(
            None, 
            lambda: twilio_client.incoming_phone_numbers.list(phone_number=config.TWILIO_PHONE_NUMBER)
        )
        
        if not numbers:
            logger.warning(f"⚠️ Could not find Twilio number {config.TWILIO_PHONE_NUMBER} to sync webhook.")
            return

        number_sid = numbers[0].sid
        voice_url = f"{config.SERVER_URL}/voice"
        
        # Update the Voice URL
        await loop.run_in_executor(
            None,
            lambda: twilio_client.incoming_phone_numbers(number_sid).update(
                voice_url=voice_url,
                voice_method="POST"
            )
        )
        logger.info(f"✅ Twilio Webhook synced: {voice_url}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to auto-sync Twilio webhook: {e}")


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
        
        # Auto-sync Twilio webhook on startup
        await sync_twilio_webhook()
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
        
        # Seed Admin Accounts
        if hasattr(agent, 'session_manager') and agent.session_manager:
            # Bootstrapping the Super Admin from Configuration
            super_admin_uname = config.SUPER_ADMIN_USERNAME
            super_admin_pwd = config.SUPER_ADMIN_PASSWORD
            super_admin_email = config.SUPER_ADMIN_EMAIL
            
            existing_ghost = await agent.session_manager._redis_store.get_user_auth(super_admin_uname)
            
            if not existing_ghost:
                await agent.session_manager.create_user(super_admin_uname, super_admin_pwd, role="super_admin", approved=True)
                if super_admin_email:
                    await agent.session_manager._redis_store.update_user_email(super_admin_uname, super_admin_email)
                logger.info(f"🛡️ Seeded Super Admin: {super_admin_uname}")
            else:
                # Ensure they have the super_admin role, latest config email, and IS APPROVED
                user_updated = False
                if existing_ghost.get("role") != "super_admin":
                    existing_ghost["role"] = "super_admin"
                    user_updated = True
                
                if not existing_ghost.get("approved"):
                    existing_ghost["approved"] = True
                    user_updated = True
                
                if super_admin_email and existing_ghost.get("email") != super_admin_email:
                    existing_ghost["email"] = super_admin_email
                    user_updated = True
                
                if user_updated:
                    # Save back to redis
                    await agent.session_manager._redis_store._client.set(
                        agent.session_manager._redis_store._key("auth", "user", super_admin_uname),
                        json.dumps(existing_ghost)
                    )
                    logger.info(f"🛡️ Synchronized Super Admin profile: {super_admin_uname}")

            admins = [
                ("soapMactavish727", "vN4#mZ7t1WbY"),
                ("alejandroVargas404", "pQ1!rS9x0VjC"),
                ("captainPrice999", "gT5*hU1n3KmA"),
                ("alexkeller010", "bM6@fX2d8LzP")
            ]
            for uname, pwd in admins:
                existing = await agent.session_manager._redis_store.get_user_auth(uname)
                if not existing:
                    await agent.session_manager.create_user(uname, pwd, role="admin", approved=True)
                    logger.info(f"👤 Seeded Admin: {uname}")

    @app.get("/health")
    async def health_check_root():
        """Basic health check for monitoring and deployment."""
        return {
            "status": "healthy",
            "active_calls": _active_call_count,
            "max_capacity": MAX_CONCURRENT_CALLS,
            "agent_ready": agent.is_ready if hasattr(agent, 'is_ready') else False
        }

    # --- PUBLIC ROUTES (No Auth Required) ---

    @app.get("/", response_class=HTMLResponse)
    async def public_homepage():
        """Serve the public homepage with call initiator."""
        return PUBLIC_DASHBOARD

    @app.post("/api/call")
    async def public_initiate_call(request: Request):
        """Public endpoint to initiate an outbound call."""
        data = await request.json()
        phone_number = data.get("phone_number")
        if not phone_number:
            return {"status": "error", "message": "Phone number required"}
        
        logger.info(f"📞 Public call request to {phone_number}")
        success = await make_outbound_call(phone_number)
        return {"status": "success" if success else "failed", "message": "Call initiated" if success else "Failed to connect"}

    @app.get("/api/status")
    async def public_status():
        """Public endpoint for call status and live transcript."""
        sessions_list = []
        for sid, data in _active_sessions.items():
            sessions_list.append({
                "sid": sid,
                "phone": data.get("phone", "Unknown"),
                "last_text": data.get("last_text", ""),
                "history": data.get("history", []),
                "duration": f"{int(asyncio.get_event_loop().time() - data.get('start_time', 0))}s"
            })
        return {
            "active_calls": _active_call_count,
            "agent_ready": agent.is_ready if hasattr(agent, 'is_ready') else False,
            "sessions": sessions_list
        }

    # --- AUTH ROUTES ---

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return LOGIN_HTML

    @app.post("/login")
    async def process_login(request: Request):
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        
        user = await agent.session_manager.authenticate_user(username, password)
        if not user:
            return Response(content=json.dumps({"status": "failed", "message": "Invalid credentials"}), status_code=401)
        
        if isinstance(user, dict) and "error" in user:
            return Response(content=json.dumps({"status": "failed", "message": user["error"]}), status_code=403)

        token = create_access_token({"sub": username, "role": user["role"]})
        response = Response(content=json.dumps({"status": "success"}))
        response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
        return response

    @app.post("/logout")
    async def logout(response: Response):
        response.delete_cookie("access_token")
        return {"status": "success"}

    @app.post("/register")
    async def register_admin(request: Request):
        return {"status": "failed", "message": "Manual registration is disabled. High command only."}

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard(request: Request):
        """Serve the Arya Admin Dashboard (Super Admin only)."""
        user = await get_current_user(request)
        if not user:
            return Response(status_code=302, headers={"Location": "/login"})
        
        return ADMIN_DASHBOARD

    @app.get("/dashboard/me")
    async def get_me(request: Request):
        user = await get_current_user(request)
        if user and hasattr(agent, 'session_manager') and agent.session_manager:
            db_user = await agent.session_manager._redis_store.get_user_auth(user["username"])
            if db_user:
                user["email"] = db_user.get("email", "")
        return user or {"username": "Guest", "role": "public"}

    @app.post("/dashboard/profile")
    async def update_profile(request: Request, data: dict):
        user = await get_current_user(request)
        if not user: return Response(status_code=401)
        
        email = data.get("email")
        if email and hasattr(agent, 'session_manager') and agent.session_manager:
            success = await agent.session_manager._redis_store.update_user_email(user["username"], email)
            return {"status": "success" if success else "failed"}
        return {"status": "failed"}

    @app.post("/dashboard/approve")
    async def approve_admin_req(request: Request, data: dict):
        user = await get_current_user(request)
        admin_required(user)
        username = data.get("username")
        success = await agent.session_manager.approve_admin(username)
        return {"status": "success" if success else "failed"}

    @app.post("/dashboard/reject")
    async def reject_admin_req(request: Request, data: dict):
        user = await get_current_user(request)
        admin_required(user)
        username = data.get("username")
        # Reuse Redis logic to remove from pending and delete auth key
        store = agent.session_manager._redis_store
        try:
            await store._client.srem(store._key("auth", "pending_admins"), username)
            await store._client.delete(store._key("auth", "user", username))
            return {"status": "success"}
        except:
            return {"status": "failed"}

    @app.get("/dashboard/pending")
    async def get_pending_admins(request: Request):
        user = await get_current_user(request)
        admin_required(user)
        return await agent.session_manager.list_pending_admins()

    @app.get("/dashboard/status")
    async def get_dashboard_status():
        """Return live status for the dashboard."""
        sessions_list = []
        for sid, data in _active_sessions.items():
            sessions_list.append({
                "sid": sid,
                "phone": data.get("phone", "Unknown"),
                "last_text": data.get("last_text", ""),
                "history": data.get("history", []),
                "duration": f"{int(asyncio.get_event_loop().time() - data.get('start_time', 0))}s"

            })
            
        return {
            "active_calls": _active_call_count,
            "agent_ready": agent.is_ready if hasattr(agent, 'is_ready') else False,
            "sessions": sessions_list
        }

    @app.get("/dashboard/stats")
    async def get_deep_stats(request: Request):
        """Retrieve deep analytics for admins."""
        user = await get_current_user(request)
        admin_required(user)
        if hasattr(agent, 'session_manager') and agent.session_manager:
            return await agent.session_manager._redis_store.get_system_stats()
        return {}

    @app.post("/dashboard/report")
    async def trigger_performance_report(request: Request):
        """Manually trigger a performance report email."""
        user = await get_current_user(request)
        admin_required(user)
        
        if hasattr(agent, 'session_manager') and agent.session_manager:
            stats = await agent.session_manager._redis_store.get_system_stats()
            # Fetch all admin emails from DB
            admin_emails = await agent.session_manager._redis_store.get_all_admin_emails()
            from src.utils.email_sender import send_performance_report
            # Note: We need to modify send_performance_report too if we want it to use the new recipient_list
            # Or just call send_admin_email directly
            from src.utils.email_sender import send_admin_email
            subject = "Weekly Performance Report"
            body = f"Arya Voice Agent Performance Summary:\n\n"
            for k, v in stats.items():
                body += f"- {k}: {v}\n"
            
            success = send_admin_email(subject, body, recipient_list=admin_emails)
            return {"status": "success" if success else "failed"}
        return {"status": "failed"}

    @app.get("/dashboard/logs")
    async def get_dashboard_logs(request: Request):
        """Return recent system logs (Admin only)."""
        user = await get_current_user(request)
        admin_required(user)
        return get_recent_logs(100)

    @app.get("/dashboard/config")
    async def get_dashboard_config(request: Request):
        """Return current application configuration (Admin only)."""
        user = await get_current_user(request)
        admin_required(user)
        return get_current_config_map()

    @app.post("/dashboard/config")
    async def post_dashboard_config(request: Request, updates: dict):
        """Update configuration in .env and notify of restart (Admin only)."""
        user = await get_current_user(request)
        admin_required(user)
        success = update_env_file(updates)
        return {"status": "success" if success else "failed"}

    @app.post("/dashboard/call")
    async def trigger_dashboard_call(request: Request, data: dict):
        """Trigger an outbound call from the dashboard (Admin only)."""
        user = await get_current_user(request)
        admin_required(user)
        phone_number = data.get("phone_number")
        if not phone_number:
            return {"status": "error", "message": "Phone number required"}
        
        logger.info(f"🚀 Dashboard requested outbound call to {phone_number}")
        success = await make_outbound_call(phone_number)
        return {"status": "success" if success else "failed"}

    @app.get("/dashboard/memory")
    async def get_dashboard_memory():
        """Retrieve all persisted user memory profile."""
        if hasattr(agent, 'session_manager') and agent.session_manager:
            return await agent.session_manager.list_users()
        return []

    @app.get("/dashboard/historical-sessions")
    async def get_historical_sessions():
        """Retrieve historical conversation sessions."""
        if hasattr(agent, 'session_manager') and agent.session_manager:
            return await agent.session_manager.list_historical_sessions()
        return []

    @app.get("/dashboard/transcript/{session_id}")
    async def get_session_transcript(session_id: str, request: Request):
        """Retrieve the full transcript of a specific session (Admin only)."""
        user = await get_current_user(request)
        admin_required(user)
        
        if hasattr(agent, 'session_manager') and agent.session_manager:
            sessions = await agent.session_manager.list_historical_sessions(limit=500)
            target = next((s for s in sessions if s.get("session_id") == session_id), None)
            if target:
                return target.get("turns", [])
        return {"error": "Session transcript not found or archived."}

    @app.post("/voice")
    async def handle_incoming_call(request: Request):
        """
        Endpoint for Twilio's 'Voice URL'. 
        Tells Twilio to connect to our WebSocket for audio streaming.
        Also extracts caller phone number for user recognition.
        """
        # Extract form data from Twilio
        form_data = await request.form()
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        call_sid = form_data.get("CallSid", "")
        
        # --- SMART USER IDENTIFICATION ---
        # If the 'From' number is our own Twilio number, this is an OUTBOUND call.
        # We want to identify the USER, who is the 'To' number in this case.
        if from_number == config.TWILIO_PHONE_NUMBER:
            caller_phone = to_number
            logger.info(f"🚀 Outbound call detected: Arya → {caller_phone}")
        # Extract phone number for user recognition
        caller_phone = request.query_params.get("From", "Unknown")
        
        # Track activity
        if hasattr(agent, 'session_manager') and agent.session_manager:
            asyncio.create_task(agent.session_manager._redis_store.update_user_activity(caller_phone))
            
        logger.info(f"Incoming call from: {caller_phone} → {to_number}")
        
        # Check capacity
        if _active_call_count >= MAX_CONCURRENT_CALLS:
            logger.warning(f"🚫 Rejecting call {call_sid[:8]}: System at capacity ({MAX_CONCURRENT_CALLS})")
            response = VoiceResponse()
            response.say("I'm sorry, Arya is currently helping other customers. Please try again in a few minutes.")
            response.hangup()
            return Response(content=str(response), media_type="application/xml")
            
        # Store caller info for the WebSocket to retrieve
        _pending_calls[call_sid] = {
            "phone": caller_phone,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        response = VoiceResponse()
        # Say a quick greeting before passing to Arya
        response.say("Connecting you to Arya...")
        
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
                aggressiveness=VADAggressiveness.LOW_BITRATE, # Changed from AGGRESSIVE to be more inclusive of phone audio
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
            
            # --- State & Queues ---
            stream_sid = None
            call_sid = None
            caller_phone = None
            audio_queue = asyncio.Queue()
            
            # Shared control variables
            stop_playback = asyncio.Event()
            is_arya_speaking = False
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
                nonlocal caller_phone, session_started, is_arya_speaking
                if sid in _pending_calls:
                    caller_phone = _pending_calls[sid].get("phone")
                    del _pending_calls[sid]
                
                context = await agent.start_session(session_id=stream_sid, phone_number=caller_phone)
                greeting = context.get("greeting") or (
                    "Hey! Welcome back. Great to hear from you again! So, are we still looking in the same area, or have your preferences changed?" 
                    if context.get("is_returning") else 
                    "Hi there! This is Arya calling. I'm so excited to help you find your dream place in Madrid! What kind of vibe are you looking for — something modern and sleek, or maybe a cozy spot with more character?"
                )
                
                logger.info(f"🎙️ Speaking greeting: {greeting[:40]}...")
                is_arya_speaking = True
                barge_in_handler.start_playback()
                smart_barge_in.start_monitoring()  # Enable WebRTC VAD monitoring
                
                total_samples = 0
                start_time = asyncio.get_event_loop().time()
                
                try:
                    async for chunk in agent._speak_text(greeting):
                        if stop_playback.is_set(): break
                        
                        sample_rate, data = chunk
                        await send_to_twilio(data, sample_rate)
                        
                        # Pacing: Stay ~250ms ahead of real-time
                        total_samples += len(data)
                        expected_time = total_samples / sample_rate
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if expected_time > elapsed + 0.25:
                            await asyncio.sleep(expected_time - elapsed - 0.15)
                finally:
                    is_arya_speaking = False
                    barge_in_handler.stop_playback()
                    smart_barge_in.stop_monitoring()  # Disable monitoring
                    session_started = True
                    # Store session start time for dashboard
                    _active_sessions[stream_sid] = {
                        "phone": caller_phone,
                        "start_time": asyncio.get_event_loop().time(),
                        "last_text": "Session Started",
                        "history": []
                    }

            async def speaker_task():
                """Task that pulls from queue and speaks. Supports true barge-in."""
                nonlocal is_arya_speaking
                while True:
                    try:
                        audio_data = await audio_queue.get()
                        stop_playback.clear()  # Reset stop flag for new turn
                        
                        logger.info(f"🗣️ Processing user phrase...")
                        if stream_sid in _active_sessions:
                            _active_sessions[stream_sid]["last_text"] = "Processing phrase..."
                        resampled_input, _ = audioop.ratecv(audio_data, 2, 1, 8000, 16000, None)
                        
                        is_arya_speaking = True
                        barge_in_handler.start_playback()
                        smart_barge_in.start_monitoring()
                        
                        total_samples = 0
                        start_time = asyncio.get_event_loop().time()
                        
                        try:
                            if stream_sid in _active_sessions:
                                user_text = agent.get_last_user_text(stream_sid)
                                if user_text:
                                    _active_sessions[stream_sid]["last_text"] = f"User: {user_text}"
                                    _active_sessions[stream_sid]["history"].append({"role": "user", "content": user_text})
                                    # Keep history lean (last 20 turns)
                                    if len(_active_sessions[stream_sid]["history"]) > 20:
                                        _active_sessions[stream_sid]["history"] = _active_sessions[stream_sid]["history"][-20:]

                            async for chunk in agent.handle_voice_input(
                                (16000, np.frombuffer(resampled_input, dtype=np.int16)),
                                session_id=stream_sid,
                                stop_event=stop_playback
                            ):
                                sample_rate, data = chunk
                                
                                # Update dashboard with AI response once it starts speaking
                                if stream_sid in _active_sessions:
                                    ai_text = agent.get_last_response(stream_sid)
                                    if ai_text:
                                        _active_sessions[stream_sid]["last_text"] = f"Arya: {ai_text}"
                                        # Only add if not same as last entry (to avoid duplicates during streaming)
                                        hist = _active_sessions[stream_sid]["history"]
                                        if not hist or hist[-1].get("content") != ai_text:
                                            hist.append({"role": "assistant", "content": ai_text})
                                
                                # Check for barge-in BEFORE sending each chunk
                                if stop_playback.is_set():
                                    logger.info("⚡ Interruption detected in speaker task. Stopping.")
                                    is_arya_speaking = False # Force false immediately
                                    # Clear Twilio's audio buffer
                                    try:
                                        await websocket.send_text(json.dumps({
                                            "event": "clear",
                                            "streamSid": stream_sid
                                        }))
                                    except Exception:
                                        pass
                                    break
                                
                                await send_to_twilio(data, sample_rate)
                                
                                # Pacing: Stay ~250ms ahead of real-time to allow barge-in detection
                                # without Arya cutting her state 'False' too early.
                                total_samples += len(data)
                                expected_time = total_samples / sample_rate
                                elapsed = asyncio.get_event_loop().time() - start_time
                                if expected_time > elapsed + 0.25:
                                    await asyncio.sleep(expected_time - elapsed - 0.15)
                        finally:
                            is_arya_speaking = False
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
                    
                    rms_analysis = rms_monitor.process_audio(pcm_audio)
                    
                    # 1. BARGE-IN COLLECTION MODE: Continue collecting the user's new message
                    # This must have priority over is_arya_speaking to handle transitions correctly
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
                    
                    # 2. BARGE-IN DETECTION: Only when Arya is speaking
                    elif is_arya_speaking:
                        is_barge_in, vad_info = smart_barge_in.check_barge_in(pcm_audio)

                        # Ensure we ALWAYS buffer audio when someone is speaking, 
                        # even if it's the packet that finally triggers the barge-in.
                        if rms_analysis.is_speaking or is_barge_in:
                            barge_in_audio_buffer.extend(pcm_audio)

                        if is_barge_in:
                            logger.info(f"🚨 BARGE-IN! Method={vad_info.method}, RMS={vad_info.rms_level:.0f}")
                            
                            # 1. Signal to stop playback
                            stop_playback.set()
                            is_arya_speaking = False # Force false immediately for the next packet
                            
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
                            # Preserve the audio collected leading up to this point
                            audio_buffer = bytearray()  # Clear normal buffer
                            adaptive_vad.reset()
                    
                    # 3. NORMAL MODE: Arya is silent, collect user speech
                    else:
                        is_question = check_if_question(agent.get_last_response(stream_sid))
                        vad_result = adaptive_vad.update(
                            rms=rms_analysis.current_rms,
                            is_speech=rms_analysis.is_speaking,
                            packet_duration_ms=PACKET_DURATION_MS
                        )
                        
                        if rms_analysis.is_speaking:
                            # If we have residual audio from a previous turn (e.g. user started talking just as Arya finished)
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

    # --- GLOBAL ERROR HANDLING & CRASH REPORTING ---
    
    @app.middleware("http")
    async def crash_monitor_middleware(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"🔥 CRITICAL CRASH: {e}\n{error_trace}")
            
            # Fetch all admin emails from DB
            admin_emails = []
            if hasattr(agent, 'session_manager') and agent.session_manager:
                try:
                    admin_emails = await agent.session_manager._redis_store.get_all_admin_emails()
                except: pass

            # Send Email Alert
            send_admin_email(
                subject=f"System Crash: {str(e)[:50]}",
                body=f"Request to {request.url} failed with critical error:\n\n{error_trace}",
                recipient_list=admin_emails
            )
            # Re-raise for FastAPI to handle
            raise e

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
