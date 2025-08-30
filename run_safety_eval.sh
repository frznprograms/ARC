#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment (if you use one)
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "Starting evaluation for Safety Model..."
uv run -m src.pipelines.safety_eval