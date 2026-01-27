#!/usr/bin/env python
"""
Verification script for Sarah Voice Agent.

Run this before deployment to ensure everything is configured correctly.

Usage:
    python verify_setup.py
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def colored(text: str, color: str) -> str:
    """Simple ANSI color wrapper."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def check_pass(msg: str):
    print(f"  {colored('✓', 'green')} {msg}")


def check_fail(msg: str):
    print(f"  {colored('✗', 'red')} {msg}")


def check_warn(msg: str):
    print(f"  {colored('!', 'yellow')} {msg}")


async def main():
    print("\n" + "=" * 60)
    print(colored("  Sarah Voice Agent - Pre-Deployment Verification", "blue"))
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # --- 1. Environment Variables ---
    print(colored("1. Environment Variables", "blue"))
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        "GROQ_API_KEY": "Required for LLM",
    }
    optional_vars = {
        "TWILIO_ACCOUNT_SID": "Required for phone calls",
        "TWILIO_AUTH_TOKEN": "Required for phone calls",
        "TWILIO_PHONE_NUMBER": "Required for phone calls",
        "SERVER_URL": "Public URL for Twilio webhooks",
        "REDIS_URL": "Persistent memory (optional)",
    }
    
    for var, desc in required_vars.items():
        if os.getenv(var):
            check_pass(f"{var}: Set")
        else:
            check_fail(f"{var}: NOT SET - {desc}")
            all_passed = False
    
    for var, desc in optional_vars.items():
        if os.getenv(var):
            check_pass(f"{var}: Set")
        else:
            check_warn(f"{var}: Not set - {desc}")
    
    # --- 2. Data Files ---
    print(f"\n{colored('2. Data Files', 'blue')}")
    
    data_path = Path(__file__).parent / "data" / "properties.csv"
    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        check_pass(f"properties.csv: {len(df)} properties loaded")
    else:
        check_fail(f"properties.csv: NOT FOUND at {data_path}")
        all_passed = False
    
    # --- 3. Module Imports ---
    print(f"\n{colored('3. Module Imports', 'blue')}")
    
    modules_to_check = [
        ("src.config", "Configuration"),
        ("src.models.llm", "LLM (Groq)"),
        ("src.models.stt", "Speech-to-Text"),
        ("src.models.tts", "Text-to-Speech"),
        ("src.tools.property_search_enhanced", "Property Search"),
        ("src.memory.session_manager", "Session Manager"),
        ("src.telephony", "Telephony (Twilio)"),
    ]
    
    for module, name in modules_to_check:
        try:
            __import__(module)
            check_pass(f"{name}: OK")
        except Exception as e:
            check_fail(f"{name}: Import error - {e}")
            all_passed = False
    
    # --- 4. WebRTC VAD ---
    print(f"\n{colored('4. Audio Processing', 'blue')}")
    
    try:
        import webrtcvad
        check_pass("WebRTC VAD: Available (ML-based speech detection)")
    except ImportError:
        check_warn("WebRTC VAD: Not installed (using RMS fallback)")
    
    try:
        import audioop
        check_pass("audioop: Available")
    except ImportError:
        check_fail("audioop: Not available")
        all_passed = False
    
    # --- 5. Redis (Optional) ---
    print(f"\n{colored('5. Redis (Optional)', 'blue')}")
    
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis.asyncio as redis_client
            r = redis_client.from_url(redis_url)
            await r.ping()
            await r.close()
            check_pass("Redis: Connected")
        except Exception as e:
            check_warn(f"Redis: Connection failed - {e}")
    else:
        check_warn("Redis: Not configured (using in-memory storage)")
    
    # --- 6. Number Formatting Test ---
    print(f"\n{colored('6. Number-to-Words Test', 'blue')}")
    
    try:
        from src.main import SarahAgent
        agent = SarahAgent()
        test_num = 500000
        result = agent._number_to_words(test_num)
        if result == "five hundred thousand":
            check_pass(f"{test_num} → '{result}'")
        else:
            check_warn(f"{test_num} → '{result}' (expected 'five hundred thousand')")
    except Exception as e:
        check_warn(f"Number conversion test failed: {e}")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    if all_passed:
        print(colored("  ✓ All critical checks passed! Ready to deploy.", "green"))
    else:
        print(colored("  ✗ Some checks failed. Please fix before deploying.", "red"))
    print("=" * 60)
    
    print(f"\n{colored('Quick Start:', 'blue')}")
    print("  Local testing:   python src/main.py")
    print("  Phone mode:      python src/main.py --phone")
    print("  Docker:          docker-compose up --build")
    print()
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
