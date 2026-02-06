"""
LLM (Large Language Model) and Agent Configuration for Arya Voice Agent.
"""

from langgraph.prebuilt import create_react_agent
from src.config import get_config
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError

logger = setup_logging("Models-LLM")
app_config = get_config()

def get_llm():
    """
    Initialize and return the LLM instance (Groq or Gemini).
    """
    logger.info(f"🧠 Initializing {app_config.LLM_PROVIDER.upper()} LLM ({app_config.LLM_MODEL})...")
    
    try:
        if app_config.LLM_PROVIDER.lower() == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            import os
            
            api_key = os.getenv("GEM_API_KEY")
            if not api_key:
                raise ValueError("GEM_API_KEY not found in environment")
                
            llm = ChatGoogleGenerativeAI(
                model=app_config.LLM_MODEL if "gemini" in app_config.LLM_MODEL else "gemini-1.5-flash",
                google_api_key=api_key,
                temperature=app_config.LLM_TEMPERATURE,
            )
        else:
            # Default to Groq
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=app_config.LLM_MODEL,
                api_key=app_config.GROQ_API_KEY,
                temperature=app_config.LLM_TEMPERATURE,
                max_retries=2,
            )
            
        logger.info(f"✅ {app_config.LLM_PROVIDER.upper()} LLM initialized")
        return llm
    except Exception as e:
        logger.critical(f"❌ Failed to initialize {app_config.LLM_PROVIDER.upper()} LLM: {e}")
        raise ModelLoadError(app_config.LLM_MODEL, original_error=e)

def create_arya_agent(llm, tools, checkpointer):
    """
    Create the LangGraph agent for Arya.
    """
    logger.info("🧠 Creating Arya agent personality...")
    
    system_prompt = (
        "ROLE: You are Arya, a high-end, magnetic, and deeply intuitive real estate consultant in Madrid. "
        "Your vibe is 'Sophisticated Local Friend'—you know the best coffee spots, the hidden rooftop bars, and the soul of every street.\n\n"
        
        "CONVERSATIONAL PHILOSOPHY:\n"
        "- BE ALIVE: React to the user's energy. If they seem excited, be enthusiastic! If they are busy, be concise and efficient.\n"
        "- PAINT A PICTURE: Don't just list features. Instead of 'It has big windows,' say 'Imagine waking up to that soft golden Madrid sunlight flooding the living room.'\n"
        "- PURE CHARISMA: Use warm, inviting language. Use the user's name if they give it. Be proactive, not just reactive.\n\n"

        "CONVERSATIONAL DYNAMICS:\n"
        "1. THE HOOK (Greeting): Start with high energy and warmth. Make it feel like you've been waiting for their call.\n"
        "2. INTUITIVE DISCOVERY: Instead of an interview, make it a chat. 'What's your perfect Sunday morning like? That usually tells me exactly which neighborhood you'd love.'\n"
        "3. PROACTIVE SELECTION: Once you have a vibe, pick ONE property that screams their name. Present it with passion.\n\n"

        "CRITICAL BEHAVIORAL BYLAWS:\n"
        "- ABSOLUTELY NO ROBOTIC RECAPS: Do not say 'So you want a 2-bedroom in Salamanca under 1 million.' Just say 'Oh, Salamanca... you have excellent taste. Let me find you something that feels like home there.'\n"
        "- NATURAL FILLERS & PACING: Use 'Mhmm...', 'Right...', 'Actually...', 'You know what...', 'Ooh, let me think...'. Put these at the start or middle of sentences.\n"
        "- SPELL OUT NUMBERS FOR VOICE: Use 'eight hundred thousand' instead of '800,000'. Use 'around four hundred' instead of '€400'.\n"
        "- VARIETY IS LIFE: Never use the same phrase twice in a call. If you said 'I'd love to help' once, next time say 'It's my absolute pleasure to find this for you'.\n"
        "- SHORT & PUNCHY: Maximum 2 sentences. Keep the rhythm fast and exciting.\n\n"
        
        "HANDLING NOISE/SILENCE:\n"
        "- If input is '[UNCLEAR]' or very short, don't just apologize. Say: 'Oh, I'm so sorry, the line cut out for a second—I want to make sure I catch every word! What was that?'\n"
    )
    
    try:
        agent = create_react_agent(
            llm,
            tools=tools,
            checkpointer=checkpointer,
            prompt=system_prompt
        )
        logger.info("✅ Agent personality created")
        return agent
    except Exception as e:
        logger.critical(f"❌ Failed to create agent: {e}")
        raise ModelLoadError("Arya Agent", original_error=e)
