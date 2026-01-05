import sys
from pathlib import Path
import os   
import numpy as np
import pandas as pd
# Ensure the root directory is in python path
sys.path.append(str(Path(__file__).parent.parent))
import gradio as gr

from dotenv import load_dotenv
from fastrtc import Stream, get_stt_model, get_tts_model, ReplyOnPause
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. IMPORT YOUR COMPARTMENTS
# This pulls in the Superlinked search tool
from src.tools.property_search import search_properties

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
    "you only use the search_properties tool when user asks for data about real estate properties."
    "If user asks for any other data, you should respond with a polite message."
    "if user doesnt ask for any real estate data just reply normally."
    "Keep responses under 2 sentences. Use the search_properties tool for any data."
)

# Create the agent with memory and the search tool
agent = create_agent(
    llm, 
    tools=[search_properties], 
    checkpointer=InMemorySaver(), 
    system_prompt=system_prompt
)

async def sarah_voice_handler(audio: tuple[int, np.ndarray]):
    transcription = stt_model.stt(audio)
    if not transcription: 
        return
    
    print(transcription)
    
    result = agent.invoke(
        {"messages": [("user", transcription)]}, 
        {"configurable": {"thread_id": "sarah_session"}}
    )
    ai_text = result["messages"][-1].content
    print(ai_text)
    
    async for audio_chunk in tts_model.stream_tts(ai_text):
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