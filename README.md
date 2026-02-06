# Arya - Real Estate Voice Agent

A voice-enabled AI agent that can chat about Madrid real estate properties over the phone. Built with **FastAPI**, **LangGraph**, **Groq**, **Edge-TTS**, and **Google Gemini**.

## 🌐 Live Demo

**Deployed Application**: [https://voiceagent-8dlo.onrender.com](https://voiceagent-8dlo.onrender.com)

Visit the live deployment to try the voice agent!

## ✨ Features
- **Cloud-First Architecture**: Minimal local footprint (~300MB).
- **Voice Recognition**: Powered by Deepgram Nova-2 (default) or Groq Whisper.
- **Voice Synthesis**: Powered by Edge-TTS (Ava US English).
- **Semantic Search**: Numpy + Google Gemini Embeddings (lightweight, no ChromaDB).
- **Natural Conversation**: Large Language Model on Groq (Llama 3.3 70B).
- **Telephony**: Twilio integration for real phone calls.
- **Persistent Memory**: Redis-backed user recognition and preferences.
- **Admin Dashboard**: Web UI for monitoring calls and configuration.

## ⚠️ Requirements

*   **Python 3.10 - 3.12**
*   **API Keys**:
    - `GROQ_API_KEY` - STT and LLM
    - `DEEPGRAM_API_KEY` - Optional: For speaker diarization
    - `GEM_API_KEY` - Property search embeddings
    - `TWILIO_*` - Phone calls

## 🚀 Setup

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd voiceagent
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # Windows
    source venv/bin/activate  # Mac/Linux
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_key
    GEM_API_KEY=your_gemini_key
    
    TWILIO_ACCOUNT_SID=your_twilio_sid
    TWILIO_AUTH_TOKEN=your_twilio_token
    TWILIO_PHONE_NUMBER=+1234567890
    
    REDIS_URL=redis://localhost:6379  # Optional
    
    SUPER_ADMIN_USERNAME=admin
    SUPER_ADMIN_PASSWORD=your_secure_password
    ```

## ▶️ Running

### 📞 Phone Mode (Recommended)
Start as a Twilio phone agent:
```bash
python -m src.main --phone
```

Then:
1. Start ngrok: `ngrok http 8000`
2. Configure Twilio webhook to your ngrok URL
3. Call your Twilio number!

### 🌐 Web Dashboard
Visit `http://localhost:8000/` for:
- **Public Homepage**: Call initiator & live transcript
- **Admin Panel** (`/admin`): Logs, configuration, intelligence

## 📁 Project Structure

```
src/
├── main.py                 # Entry point and voice handler
├── telephony.py            # Twilio WebSocket & API routes
├── dashboard_template.py   # Public & Admin UI templates
├── config.py               # Configuration management
├── models/                 # AI models (STT, TTS, LLM)
├── tools/                  # Agent tools (property search)
├── memory/                 # Redis & session management
├── audio/                  # VAD, barge-in detection
└── utils/                  # Logging, auth, embeddings
data/
└── properties.csv          # Property listings database
```

## 🔧 Architecture

```
User Call → Twilio → WebSocket → STT → LLM Agent → TTS → Twilio → User
                                  ↓
                          Property Search Tool
                          (Gemini Embeddings + Numpy)
```

## 📊 Size Optimizations

This build is optimized for minimal footprint:
- No PyTorch, TensorFlow, or heavy ML frameworks
- Cloud-based embeddings (Gemini API)
- No ChromaDB or Pandas dependency
- Total venv size: ~300MB
