# Production Dockerfile for Arya Voice Agent
# Optimized for Render.com deployment
FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Render provides PORT dynamically, default to 10000
ENV PORT=10000
EXPOSE 10000

# Start with uvicorn directly for better control
# Use --host 0.0.0.0 to bind to all interfaces (required for Render)
CMD ["sh", "-c", "python -m src.main --phone"]
