#!/bin/bash

# Exit immediately if a command fails
set -e

# Path to your venv (adjust if needed)
VENV_DIR="venv"

# Activate venv
if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/Scripts/activate"
else
  echo "❌ Virtual environment not found. Run 'python -m venv venv' first."
  exit 1
fi

# Start FastAPI backend in the background
echo "🚀 Starting FastAPI backend..."
uvicorn backend:app --host 0.0.0.0 --port 8000 &

# Save the PID so we can kill it later
FASTAPI_PID=$!

# Start Streamlit frontend
echo "🌐 Starting Streamlit app..."
streamlit run app.py --server.port 8501

# When Streamlit exits, kill FastAPI too
kill $FASTAPI_PID
