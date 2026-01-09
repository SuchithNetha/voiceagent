# Sarah - Real Estate Voice Agent

A voice-enabled AI agent that can chat about Madrid real estate properties. Built with **FastRTC**, **LangGraph**, **Groq**, and **Superlinked**.

## ⚠️ Requirements

*   **Python 3.10, 3.11, or 3.12** (Python 3.13 is NOT supported by Superlinked yet)
*   **Grok API Key**

## 🚀 Setup

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repo-url>
    cd voiceagent
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    # Ensure you are using Python 3.12 or lower
    python3.12 -m venv venv
    source venv/bin/activate  # Mac/Linux
    .\venv\Scripts\activate   # Windows
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_api_key_here
    ```

## ▶️ Running

Run the agent from the project root:

```bash
python -m src.main
```

Then open the URL shown in the console (usually `http://127.0.0.1:7860`).

## 📁 Project Structure

*   `src/main.py`: The entry point and voice handler.
*   `src/tools/property_search.py`: The "memory" engine using Superlinked.
*   `data/properties.csv`: The database of properties.
