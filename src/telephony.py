"""
Telephony Module for Sarah Voice Agent.
Handles Twilio Media Streams, Webhooks, and Outbound Calls.
"""

import json
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect

from src.config import get_config
from src.utils.logger import setup_logging

logger = setup_logging("Telephony")
config = get_config()

# Twilio Client for Outbound Calls
twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN) if config.TWILIO_ACCOUNT_SID else None

def create_telephony_app(agent):
    """
    Creates a FastAPI app to handle Twilio calls.
    """
    app = FastAPI()

    @app.post("/voice")
    async def handle_incoming_call(request: Request):
        """
        Endpoint for Twilio's 'Voice URL'. 
        Tells Twilio to connect to our WebSocket for audio streaming.
        """
        response = VoiceResponse()
        # Say a quick greeting before passing to Sarah
        response.say("Connecting you to Sarah...")
        
        connect = Connect()
        # Connect to our WebSocket stream
        stream_url = f"wss://{config.SERVER_URL.replace('https://', '').replace('http://', '')}/media-stream"
        connect.stream(url=stream_url)
        response.append(connect)
        
        logger.info("📞 Incoming call received - Connecting to Media Stream")
        return Response(content=str(response), media_type="application/xml")

    @app.websocket("/media-stream")
    async def handle_media_stream(websocket: WebSocket):
        """
        WebSocket handler for Twilio's audio stream.
        Uses manual buffering and silence detection.
        """
        import audioop
        await websocket.accept()
        logger.info("🎙️ Twilio Media Stream WebSocket connected")
        
        stream_sid = None
        audio_buffer = bytearray()
        silence_threshold = 1000  # Increased for phone noise
        silence_duration = 0      # How many consecutive silent packets
        PACKET_DURATION = 20      # Twilio sends 20ms chunks
        SILENCE_LIMIT = 800       # 800ms of silence counts as 'finished'
        
        async def send_to_twilio(audio_chunk_np: np.ndarray, source_sample_rate: int):
            """Convert Sarah's float32 audio to int16 mulaw for Twilio."""
            if not stream_sid: return
            
            try:
                # 1. Convert Sarah's float32 -> int16
                # Kokoro outputs floats between -1.0 and 1.0. 
                # We multiply by 32767 to fill the int16 range.
                if audio_chunk_np.dtype == np.float32:
                    audio_int16 = (audio_chunk_np * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_chunk_np.astype(np.int16)
                
                audio_bytes = audio_int16.tobytes()
                
                # 2. Resample from source rate -> Twilio's rate (8kHz)
                resampled, _ = audioop.ratecv(audio_bytes, 2, 1, source_sample_rate, 8000, None)
                
                # 3. Convert to mulaw and send
                mulaw_data = audioop.lin2ulaw(resampled, 2)
                payload = base64.b64encode(mulaw_data).decode('utf-8')
                
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload}
                }))
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

        try:
            while True:
                message = await websocket.receive_text()
                packet = json.loads(message)
                
                if packet['event'] == 'start':
                    stream_sid = packet['start']['streamSid']
                    logger.info(f"🚀 Stream started: {stream_sid}")
                    
                elif packet['event'] == 'media':
                    # 1. Decode Twilio Audio (8k mulaw)
                    mulaw_audio = base64.b64decode(packet['media']['payload'])
                    pcm_audio = audioop.ulaw2lin(mulaw_audio, 2)
                    
                    # 2. Check for silence using RMS (Root Mean Square)
                    rms = audioop.rms(pcm_audio, 2)
                    
                    if rms < silence_threshold:
                        silence_duration += PACKET_DURATION
                    else:
                        silence_duration = 0
                        # Only add to buffer if someone is actually talking or we haven't finished a phrase
                        audio_buffer.extend(pcm_audio)

                    # 3. If we've gathered audio and now there's silence, process it
                    if silence_duration >= SILENCE_LIMIT and len(audio_buffer) > 3200: # at least 200ms of audio
                        logger.info(f"🗣️ Phrase captured ({len(audio_buffer)//32}ms), processing...")
                        
                        # Convert buffer to 16kHz for Sarah
                        resampled_input, _ = audioop.ratecv(bytes(audio_buffer), 2, 1, 8000, 16000, None)
                        
                        # Clear buffer and silence counter BEFORE processing to allow interruptions 
                        # (though Sarah isn't full-duplex yet)
                        audio_buffer = bytearray()
                        silence_duration = 0
                        
                        async for response_chunk in agent.handle_voice_input((16000, np.frombuffer(resampled_input, dtype=np.int16))):
                            sample_rate, audio_data = response_chunk
                            # audio_data is a numpy array (float32 or int16)
                            await send_to_twilio(audio_data, sample_rate)

                elif packet['event'] == 'stop':
                    logger.info("⏹️ Stream stopped")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Media Stream error: {e}")
        finally:
            logger.info("🔌 Twilio Media Stream disconnected")

    return app

def make_outbound_call(to_number: str):
    """
    Initiate a call from Sarah to a human.
    """
    if not twilio_client:
        logger.error("❌ Twilio client not initialized - Check your .env setup")
        return False
    
    try:
        call = twilio_client.calls.create(
            from_=config.TWILIO_PHONE_NUMBER,
            to=to_number,
            url=f"{config.SERVER_URL}/voice"
        )
        logger.info(f"🚀 Outbound call initiated: {call.sid}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to make outbound call: {e}")
        return False
