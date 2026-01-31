import os
import json
from pathlib import Path
from typing import List, Dict, Any
from src.config import get_config, AppConfig

ROOT_DIR = Path(__file__).resolve().parent.parent

def get_recent_logs(n: int = 100) -> List[str]:
    """Read the last n lines from the log file."""
    log_file = ROOT_DIR / "logs" / "arya.log"
    if not log_file.exists():

        return ["Log file not found."]
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            # Simple way to get last N lines without reading everything
            # For a production app, we'd use a more efficient tail-like approach
            lines = f.readlines()
            return lines[-n:]
    except Exception as e:
        return [f"Error reading logs: {e}"]

def get_current_config_map() -> Dict[str, Any]:
    """Get the current configuration as a flattened dictionary for the UI."""
    config = get_config()
    # Mask sensitive keys
    def mask(val):
        if not val or not isinstance(val, str): return val
        if len(val) < 8: return "****"
        return val[:4] + "..." + val[-4:]

    return {
        "GROQ_API_KEY": mask(config.GROQ_API_KEY),
        "TWILIO_ACCOUNT_SID": mask(config.TWILIO_ACCOUNT_SID),
        "TWILIO_AUTH_TOKEN": mask(config.TWILIO_AUTH_TOKEN),
        "TWILIO_PHONE_NUMBER": config.TWILIO_PHONE_NUMBER,
        "SERVER_URL": config.SERVER_URL,
        "LLM_MODEL": config.LLM_MODEL,
        "LLM_TEMPERATURE": config.LLM_TEMPERATURE,
        "STT_MODEL": config.STT_MODEL,
        "TTS_MODEL": config.TTS_MODEL,
        "MEMORY_MAX_TURNS": config.MEMORY_MAX_TURNS,
        "RMS_BARGE_IN_THRESHOLD": config.RMS_BARGE_IN_THRESHOLD,
    }

def update_env_file(updates: Dict[str, str]) -> bool:
    """Update the .env file with new values."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        # Create it if it doesn't exist
        with open(env_path, "w") as f:
            for k, v in updates.items():
                f.write(f"{k}={v}\n")
        return True

    try:
        with open(env_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        keys_to_update = set(updates.keys())
        
        for line in lines:
            if "=" in line and not line.startswith("#"):
                key = line.split("=")[0].strip()
                if key in keys_to_update:
                    new_lines.append(f"{key}={updates[key]}\n")
                    keys_to_update.remove(key)
                    continue
            new_lines.append(line)
        
        # Append new keys
        for key in keys_to_update:
            new_lines.append(f"{key}={updates[key]}\n")

        with open(env_path, "w") as f:
            f.writelines(new_lines)
        
        return True
    except Exception as e:
        print(f"Error updating .env: {e}")
        return False
