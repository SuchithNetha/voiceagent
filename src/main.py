import sys
from pathlib import Path
import os   
import numpy as np
import pandas as pd
# Ensure the root directory is in python path
sys.path.append(str(Path(__file__).parent.parent))
import gradio as gr
from src.tools.property_search import search_properties
from dotenv import load_dotenv
from fastrtc import Stream, get_stt_model, get_tts_model, ReplyOnPause
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import warnings

# Suppress annoying Pydantic warnings from Superlinked/Site Packages
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")


# 1. IMPORT YOUR COMPARTMENTS
# This pulls in the Superlinked search tool
load_dotenv()

# 2. INITIALIZE THE ENGINES
# (Data loading is handled inside the tool)

# Free local models (FastRTC defaults)
stt_model = get_stt_model()
tts_model = get_tts_model()

# 3. SET UP THE AGENT (THE BRAIN)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

system_prompt = (
    "You are Sarah, a warm real estate agent in Madrid. "
    "If you need to search for properties, ALWAYS start your message with a short filler "
    "like 'Let me look that up for you' or 'Sure, checking the database now'. "
    "NEVER speak function names or JSON. Only speak natural human sentences."
    "Start describing the property details as soon as you say 'let me look that up for you' after user asks for detail"
)
# Create the agent with memory and the search tool
agent = create_agent(
    llm, 
    tools=[search_properties], 
    checkpointer=InMemorySaver(), 
    system_prompt=system_prompt
)

import re

async def sarah_voice_handler(audio: tuple[int, np.ndarray]):
    transcription = stt_model.stt(audio)
    if not transcription: 
        return
    
    print(f"\n[USER]: {transcription}")
    
    # --- STEP 1: INITIAL THOUGHT ---
    # We use agent.invoke to see if she needs a tool.
    result = agent.invoke(
        {"messages": [("user", transcription)]}, 
        {"configurable": {"thread_id": "sarah_session"}}
    )
    
    last_msg = result["messages"][-1]

    # --- STEP 2: SILENT DATA SEARCH ---
    # Check if the AI wants to use Superlinked
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print(f"🛠️  [DEBUG]: LLM is searching Superlinked for: {last_msg.tool_calls[0]['args']}")
        
        # 1. Run the search tool (INTACT - No changes to your tool)
        query = last_msg.tool_calls[0]['args'].get("user_request", "")
        search_data = search_properties(query)
        
        # 2. FEEDBACK: Give the results back to the AI immediately
        # This stops the 'loop' because the AI now HAS the answer.
        final_response = agent.invoke(
            {
                "messages": [
                    last_msg, 
                    {
                        "role": "tool", 
                        "content": str(search_data), 
                        "tool_call_id": last_msg.tool_calls[0]['id']
                    }
                ]
            },
            {"configurable": {"thread_id": "sarah_session"}}
        )
        ai_text = final_response["messages"][-1].content
    else:
        # No tool was needed (just a 'Hello' etc.)
        ai_text = last_msg.content

    # --- STEP 3: THE SPEECH FILTER (KEEPING IT CLEAN) ---
    # This removes <function> tags AND any JSON { "brackets" } from the speech
    clean_text = re.sub(r'<function.*?>.*?</function>', '', ai_text)
    clean_text = re.sub(r'\{.*?\}', '', clean_text).strip()

    # --- STEP 4: INSTANT VOICE OUTPUT ---
    if clean_text:
        print(f"🎙️  [SARAH]: {clean_text}")
        # Kokoro streams the clean text immediately
        async for audio_chunk in tts_model.stream_tts(clean_text):
            yield audio_chunk
# 5. START THE WEB UI
stream = Stream(
    handler=ReplyOnPause(sarah_voice_handler), 
    modality="audio", 
    mode="send-receive"
)

if __name__ == "__main__":
    print("🏠 Sarah is online! Talk to her about Madrid real estate.")
    stream.ui.launch()