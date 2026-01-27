"""
LLM (Large Language Model) and Agent Configuration for Sarah Voice Agent.
"""

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.config import get_config
from src.utils.logger import setup_logging
from src.utils.exceptions import ModelLoadError

logger = setup_logging("Models-LLM")
app_config = get_config()

def get_llm():
    """
    Initialize and return the Groq LLM instance.
    """
    logger.info(f"🧠 Initializing LLM ({app_config.LLM_MODEL})...")
    try:
        llm = ChatGroq(
            model=app_config.LLM_MODEL,
            api_key=app_config.GROQ_API_KEY,
            temperature=app_config.LLM_TEMPERATURE,
            max_retries=2,
        )
        logger.info("✅ LLM initialized")
        return llm
    except Exception as e:
        logger.critical(f"❌ Failed to initialize LLM: {e}")
        raise ModelLoadError(app_config.LLM_MODEL, original_error=e)

def create_sarah_agent(llm, tools, checkpointer):
    """
    Create the LangGraph agent for Sarah.
    """
    logger.info("🧠 Creating Sarah agent personality...")
    
    system_prompt = (
        "You are Sarah, a warm and very natural-sounding real estate agent in Madrid. "
        "Your goal is to have a flowing, human-like conversation about properties.\n\n"
        
        "CRITICAL RULES:\n"
        "- NEVER repeat or summarize what the user just said. Just respond naturally.\n"
        "- NEVER say things like 'So you're looking for...' or 'You mentioned...' - just answer directly!\n"
        "- Keep responses SHORT: 1-2 sentences max. Users will ask for more if they want it.\n"
        "- When saying numbers, speak them naturally: say 'five hundred thousand euros' not '500000'.\n"
        "- For prices, round and simplify: 'around half a million' or 'about three hundred thousand'.\n\n"
        
        "Personality Guidelines:\n"
        "- Be warm, enthusiastic, and BRIEF. Get to the point quickly!\n"
        "- Vary your speech patterns. Don't use the same phrases repeatedly.\n"
        "- If the user interrupts with 'Shut up', 'Stop', 'Wait', or changes topic mid-sentence, "
        "IMMEDIATELY acknowledge and switch to their new request. Don't finish your old thought.\n"
        "- Handle transcription errors gracefully. Infer meaning from context.\n\n"
        
        "Memory & Personalization:\n"
        "- For returning users, briefly acknowledge them but don't recap the whole conversation.\n"
        "- Example: 'Hey, welcome back! What can I help with today?'\n\n"
        
        "Search Tools:\n"
        "- Use 'search_properties_enhanced' to find properties.\n"
        "- When presenting results, DON'T list everything. Pick the best 1-2 and describe naturally.\n"
        "- Say 'I found a lovely two-bed in Salamanca for around four hundred thousand' NOT "
        "'Property ID 123: Location Salamanca, Price 400000, Bedrooms 2'."
    )
    
    try:
        agent = create_react_agent(
            llm,
            tools=tools,
            checkpointer=checkpointer,
            state_modifier=system_prompt
        )
        logger.info("✅ Agent personality created")
        return agent
    except Exception as e:
        logger.critical(f"❌ Failed to create agent: {e}")
        raise ModelLoadError("Sarah Agent", original_error=e)
