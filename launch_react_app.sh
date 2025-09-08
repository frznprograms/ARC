#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting ARC React Frontend & Backend..."

# Function to kill background processes on exit
cleanup() {
  echo -e "\n🛑 Shutting down servers..."
  if [[ ! -z $BACKEND_PID ]]; then
    kill $BACKEND_PID 2>/dev/null
    echo "✅ Backend stopped"
  fi
  if [[ ! -z $FRONTEND_PID ]]; then
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Frontend stopped"
  fi
  exit 0
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT INT TERM

# Activate virtual environment if exists
if [[ -d ".venv" ]]; then
    echo "📦 Activating virtual environment..."
    source .venv/Scripts/activate
fi

# Start FastAPI backend in background
echo "🔧 Starting FastAPI backend..."
uv run uvicorn src.app.backend:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start React frontend
echo "⚛️  Starting React frontend..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

echo ""
echo "🌟 ARC Review Analyzer is running!"
echo "📱 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo ""
echo "💡 Don't forget to:"
echo "   1. Set NEXT_PUBLIC_GOOGLE_MAPS_API_KEY in frontend/.env.local"
echo "   2. Ensure your HuggingFace token is in the root .env file"
echo ""
echo "Press Ctrl+C to stop both servers..."

# Keep script running and wait for user interrupt
wait

