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
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


# Load environment variables from .env file in the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")



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
    HOST: str = os.getenv("HOST", "0.0.0.0")  # Bind to all interfaces for Cloud/Docker
    PORT: int = int(os.getenv("PORT", "10000")) # Render usually provides PORT
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # --- Twilio (Telephony) ---
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    SERVER_URL: str = None  # Resolved in __post_init__
    
    # --- Audio/VAD Settings ---
    # Optimized for 8kHz μ-law from Twilio (limited dynamic range)
    # μ-law decoding typically results in lower RMS values than expected
    SILENCE_THRESHOLD_MS: int = 600          # Base silence threshold (responsive)
    SILENCE_THRESHOLD_MIN_MS: int = 400      # Minimum for fast talkers
    SILENCE_THRESHOLD_MAX_MS: int = 1000     # Maximum for deliberate speech
    RMS_SILENCE_THRESHOLD: int = 250         # INCREASED: To be less sensitive to static/background noise (was 150)
    RMS_BARGE_IN_THRESHOLD: int = 450        # INCREASED: Require clearer, louder speech to trigger barge-in (was 200)
    BARGE_IN_CONFIRM_FRAMES: int = 5          # INCREASED: Require 100ms of speech to confirm interruption (was 3)
    BARGE_IN_GRACE_PERIOD_MS: int = 500       # Give agent a breath before allowing interruption
    
    # --- Memory Settings ---
    MEMORY_MAX_TURNS: int = 5                # Max conversation turns in context
    MEMORY_SUMMARY_INTERVAL: int = 10        # Turns between auto-summarization
    MEMORY_PERSISTENCE: bool = False         # Enable file-based storage
    MEMORY_STORAGE_PATH: Optional[Path] = None
    
    # --- Search Settings ---
    SEARCH_DESCRIPTION_WEIGHT: float = 0.40  # Weight for text similarity
    SEARCH_CHARM_WEIGHT: float = 0.10        # Weight for charm preference
    SEARCH_MODERN_WEIGHT: float = 0.10       # Weight for modern preference
    SEARCH_LUXURY_WEIGHT: float = 0.10       # Weight for luxury preference
    SEARCH_PRICE_WEIGHT: float = 0.20        # Weight for price preference
    SEARCH_RESULT_LIMIT: int = 3             # Default number of results
    
    def __post_init__(self):
        # Resolve SERVER_URL logic
        # Priority:
        # 1. RENDER_EXTERNAL_URL (when deployed on Render)
        # 2. SERVER_URL from .env (if it's a valid public URL, not ngrok on Render)
        # 3. Auto-detect ngrok (when running locally)
        # 4. Default to localhost
        
        env_url = os.getenv("SERVER_URL", "")
        render_url = os.getenv("RENDER_EXTERNAL_URL", "")
        
        # On Render: ALWAYS use RENDER_EXTERNAL_URL (ngrok doesn't work there!)
        if render_url:
            self.SERVER_URL = render_url
        # Not on Render: Check if SERVER_URL is valid
        elif env_url and "ngrok" not in env_url and "localhost" not in env_url and "127.0.0.1" not in env_url:
            # SERVER_URL is a proper public URL (not ngrok, not localhost)
            self.SERVER_URL = env_url
        elif env_url and ("ngrok" in env_url or "localhost" in env_url or "127.0.0.1" in env_url):
            # SERVER_URL is ngrok or localhost - will try to auto-detect in validate()
            self.SERVER_URL = env_url
        else:
            self.SERVER_URL = "http://127.0.0.1:10000"

        """Set computed paths after initialization."""
        if self.DATA_PATH is None:
            self.DATA_PATH = self.BASE_DIR / "data" / "properties.csv"
        if self.AVATAR_DIR is None:
            self.AVATAR_DIR = self.BASE_DIR / "avatars"
        if self.LOG_DIR is None:
            self.LOG_DIR = self.BASE_DIR / "logs"
        if self.MEMORY_STORAGE_PATH is None:
            self.MEMORY_STORAGE_PATH = self.BASE_DIR / "data" / "memory"
        
        # Super Admin Defaults
        self.SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME", "simonriley141")
        self.SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD", "Rt8$kL2p9MxQ")
        self.SUPER_ADMIN_EMAIL = os.getenv("SUPER_ADMIN_EMAIL")
    
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
        
        from src.utils.logger import setup_logging
        logger = setup_logging("Config")
        
        # Check if we're on Render
        render_url = os.getenv("RENDER_EXTERNAL_URL", "")
        if render_url:
            logger.info(f"☁️ Running on Render: {self.SERVER_URL}")
            # No need to auto-detect ngrok on Render
            return
        
        # Try to auto-detect ngrok URL if we're on localhost or have a stale ngrok URL
        should_detect = (
            "127.0.0.1" in self.SERVER_URL or 
            "localhost" in self.SERVER_URL or
            "ngrok" in self.SERVER_URL
        )
        
        if should_detect:
            detected_url = self.auto_detect_server_url()
            if detected_url:
                if detected_url != self.SERVER_URL:
                    logger.info(f"🌐 Auto-detected ngrok URL: {detected_url}")
                    if "ngrok" in self.SERVER_URL and self.SERVER_URL != detected_url:
                        logger.warning(f"⚠️ Replacing stale ngrok URL: {self.SERVER_URL}")
                    self.SERVER_URL = detected_url
                else:
                    logger.info(f"✅ Ngrok URL confirmed: {detected_url}")
            elif "ngrok" in self.SERVER_URL:
                logger.warning(f"⚠️ Ngrok not detected locally, but SERVER_URL has ngrok: {self.SERVER_URL}")
                logger.warning("   If running locally, make sure ngrok is running: ngrok http 10000")
        
        logger.info(f"🔗 Final SERVER_URL: {self.SERVER_URL}")

    def auto_detect_server_url(self) -> Optional[str]:
        """
        Attempt to automatically find the public ngrok URL via its local API.
        This removes the need to manually update the .env file.
        """
        try:
            # Ngrok's local API usually lives on port 4040
            response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=1)
            if response.status_code == 200:
                tunnels = response.json().get("tunnels", [])
                for tunnel in tunnels:
                    # Look for the https tunnel
                    if tunnel.get("proto") == "https":
                        return tunnel.get("public_url")
        except Exception:
            # Ngrok probably isn't running or API isn't enabled
            pass
        return None
    
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
            
            HOST=os.getenv("HOST", "0.0.0.0"),
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