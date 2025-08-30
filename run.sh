#!/bin/bash
set -e

# If no args given → run with defaults
if [ $# -eq 0 ]; then
  source .venv/bin/activate
  python main.py
else
  # Forward all args directly to main.py
  source .venv/bin/activate
  python main.py "$@"
fi