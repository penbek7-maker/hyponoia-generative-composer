#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display alert "Python 3 is required" message "Install Python 3.10, 3.11 or 3.12, then run Install Hyponoia again."'
  exit 1
fi

python3 -m venv .venv
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements-app.txt

osascript -e 'display dialog "Hyponoia is installed. Double-click Open Hyponoia.command to begin." buttons {"OK"} default button "OK"'
