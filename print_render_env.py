import os
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# -----------------------------------------------------------
# Setup Guide for Render Deployment
# -----------------------------------------------------------

print("\n🚀 ARYA VOICE AGENT - DEPLOYMENT HELPER\n")
print("Since you are deploying to Render, you need to manually set your Environment Variables.")
print("The .env file is gitignored for security, so Render cannot see it automatically.")
print("\n📋 COPY AND PASTE THESE VALUES INTO RENDER DASHBOARD (Environment Tab):\n")

env_vars = {
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "your_key_here"),
    "GEM_API_KEY": os.getenv("GEM_API_KEY", "your_key_here"),
    "SARVAM_API_KEY": os.getenv("SARVAM_API_KEY", "your_key_here"),
    "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID", "your_sid_here"),
    "TWILIO_AUTH_TOKEN": os.getenv("TWILIO_AUTH_TOKEN", "your_token_here"),
    "TWILIO_PHONE_NUMBER": os.getenv("TWILIO_PHONE_NUMBER", "your_number_here"),
    "REDIS_URL": os.getenv("REDIS_URL", "your_redis_url_here"),
    "STT_MODEL": "sarvam",
    "TTS_MODEL": "sarvam",
    "LLM_MODEL": "llama-3.3-70b-versatile",
    # Render automatically sets RENDER_EXTERNAL_URL
}

for key, value in env_vars.items():
    safe_val = value if value else "MISSING"
    print(f"{key}={safe_val}")

print("\n\n✅ INSTRUCTIONS:")
print("1. Go to your Render Dashboard -> Select 'arya-voice-agent' Service")
print("2. Click on 'Environment' -> 'Add Environment Variable'")
print("3. Copy-paste the keys and values from above.")
print("4. Redeploy.")
