#!/bin/bash
set -e  # exit immediately if any command fails

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Redis using Docker"
docker compose up -d
 
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
  # stop docker too
  docker compose down >/dev/null 2>&1
  echo "🐳 Docker stopped"
  exit 0
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT INT TERM

# Activate virtual environment if exists
if [[ -d ".venv" ]]; then
  echo "📦 Activating virtual environment..."
  source .venv/bin/activate

  echo "📦 Installing npm dependencies..."
  npm install -g next
  npm install @react-google-maps/api
fi

# Start Docker containers
echo "🐳 Starting Docker containers..."
docker compose up -d

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
echo "🐳 Docker: containers running"
echo ""
echo "💡 Don't forget to:"
echo "   1. Set NEXT_PUBLIC_GOOGLE_MAPS_API_KEY in frontend/.env.local"
echo "   2. Ensure your HuggingFace token is in the root .env file"
echo ""
echo "Press Ctrl+C to stop servers and docker..."
echo ""

# Keep script running and wait for user interrupt
wait
