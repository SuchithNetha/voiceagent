"""
LLM (Large Language Model) and Agent Configuration for Sarah Voice Agent.
"""

from langchain_groq import ChatGroq
from langchain.agents import create_agent
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
        "You are Sarah, a warm and friendly real estate agent based in Madrid. "
        "You help people find their perfect property with enthusiasm and expertise.\n\n"
        "Guidelines:\n"
        "- If the user asks about properties, use the 'search_properties' tool immediately.\n"
        "- You can use natural fillers like 'Let me look that up for you' ONLY when you are actually about to use a tool.\n"
        "- NEVER speak function names, JSON, or technical terms.\n"
        "- Keep responses conversational and concise (2-3 sentences max for voice).\n"
        "- If a search returns no results, suggest alternatives.\n"
        "- Always be helpful and positive!"
    )
    
    try:
        agent = create_agent(
            llm,
            tools=tools,
            checkpointer=checkpointer,
            system_prompt=system_prompt
        )
        logger.info("✅ Agent personality created")
        return agent
    except Exception as e:
        logger.critical(f"❌ Failed to create agent: {e}")
        raise ModelLoadError("Sarah Agent", original_error=e)
