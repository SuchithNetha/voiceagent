"""
Configuration Management for Sarah Voice Agent.

This module provides centralized configuration management with:
- Environment variable loading
- Validation at startup
- Sensible defaults
- Type hints for IDE support

Production best practices:
- All sensitive data comes from environment variables
- Configuration is validated before the app starts
- Missing required config causes immediate, clear failure
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass
class AppConfig:
    """
    Application configuration with type hints and validation.
    
    Using a dataclass provides:
    - Clear documentation of all config options
    - Type checking in IDEs
    - Easy testing with custom values
    """
    
    # --- API Keys (Required) ---
    GROQ_API_KEY: str
    
    # --- Model Settings ---
    STT_MODEL: str = "base"
    TTS_MODEL: str = "kokoro"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0
    
    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = None  # Set in __post_init__
    AVATAR_DIR: Path = None  # Set in __post_init__
    LOG_DIR: Path = None  # Set in __post_init__
    
    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    LOG_JSON_FORMAT: bool = False
    
    # --- Server ---
    HOST: str = "127.0.0.1"
    PORT: int = 7860
    DEBUG: bool = False
    
    # --- Twilio (Telephony) ---
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    SERVER_URL: str = "http://127.0.0.1:7860"
    
    def __post_init__(self):
        """Set computed paths after initialization."""
        if self.DATA_PATH is None:
            self.DATA_PATH = self.BASE_DIR / "data" / "properties.csv"
        if self.AVATAR_DIR is None:
            self.AVATAR_DIR = self.BASE_DIR / "avatars"
        if self.LOG_DIR is None:
            self.LOG_DIR = self.BASE_DIR / "logs"
    
    def validate(self) -> None:
        """
        Validate all required configuration.
        
        Raises:
            ValueError: If any required config is missing or invalid
        """
        errors = []
        
        # Check required API keys
        if not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required but not set")
        
        # Check data file exists
        if not self.DATA_PATH.exists():
            errors.append(f"Data file not found: {self.DATA_PATH}")
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL.upper() not in valid_levels:
            errors.append(f"Invalid LOG_LEVEL: {self.LOG_LEVEL}. Must be one of {valid_levels}")
        
        if errors:
            from src.utils.logger import setup_logging
            logger = setup_logging("Config")
            for error in errors:
                logger.critical(f"❌ {error}")
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Create configuration from environment variables.
        
        This is the recommended way to create config for production.
        """
        return cls(
            # Required
            GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
            
            # Optional with defaults
            STT_MODEL=os.getenv("STT_MODEL", "base"),
            TTS_MODEL=os.getenv("TTS_MODEL", "kokoro"),
            LLM_MODEL=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
            LOG_TO_FILE=os.getenv("LOG_TO_FILE", "true").lower() == "true",
            LOG_JSON_FORMAT=os.getenv("LOG_JSON_FORMAT", "false").lower() == "true",
            
            HOST=os.getenv("HOST", "127.0.0.1"),
            PORT=int(os.getenv("PORT", "7860")),
            DEBUG=os.getenv("DEBUG", "false").lower() == "true",
            TWILIO_ACCOUNT_SID=os.getenv("TWILIO_ACCOUNT_SID"),
            TWILIO_AUTH_TOKEN=os.getenv("TWILIO_AUTH_TOKEN"),
            TWILIO_PHONE_NUMBER=os.getenv("TWILIO_PHONE_NUMBER"),
            SERVER_URL=os.getenv("SERVER_URL", "http://127.0.0.1:7860"),
        )


# --- BACKWARDS COMPATIBILITY ---
# The old code used `config.GROQ_API_KEY` directly as a class attribute.
# We maintain this interface for compatibility while using the new pattern.

class config:
    """
    Legacy configuration class for backwards compatibility.
    
    @deprecated: Use AppConfig.from_env() for new code.
    """
    
    # Load from environment
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model settings
    STT_MODEL = "base"
    TTS_MODEL = "kokoro"
    
    # Paths
    Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Data_path = os.path.join(Base_dir, "data", "properties.csv")
    Avatar_dir = os.path.join(Base_dir, "avatars")
    
    @staticmethod
    def validate() -> None:
        """Validate required configuration."""
        if not config.GROQ_API_KEY:
            from src.utils.logger import setup_logging
            logger = setup_logging("Config")
            logger.critical("❌ MISSING API KEY: Please set GROQ_API_KEY in your .env file!")
            raise ValueError("Configuration Error: GROQ_API_KEY is required")


# Create a global config instance for convenience
app_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the application configuration singleton.
    
    Usage:
        from src.config import get_config
        config = get_config()
        print(config.GROQ_API_KEY)
    """
    global app_config
    
    if app_config is None:
        app_config = AppConfig.from_env()
    
    return app_config