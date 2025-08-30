#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment (if you use one)
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Start backend (FastAPI) in background
echo "🚀 Starting FastAPI backend..."
uvicorn src.app.backend:app --reload &

# Save PID so we can kill it later
BACKEND_PID=$!

# Start frontend (Streamlit) in foreground
echo "🚀 Starting Streamlit frontend..."
streamlit run src/app/app.py

# Kill backend when Streamlit is closed
kill $BACKEND_PID