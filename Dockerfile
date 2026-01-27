# Production Dockerfile for Sarah Voice Agent
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Expose the default port (7860 is standard for Hugging Face)
EXPOSE 7860

# Metadata
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Default command: Start Sarah with telephony support
# This starts the FastAPI server which handles the Twilio Webhooks
CMD ["python", "src/main.py", "--phone"]
