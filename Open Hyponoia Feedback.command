#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -x ".venv/bin/python" ]; then
  osascript -e 'display alert "Hyponoia is not installed yet" message "Complete the installation first, then double-click this file again."'
  exit 1
fi

exec ".venv/bin/python" "hyponoia_feedback_app.py"
