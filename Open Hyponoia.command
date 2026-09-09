#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  osascript -e 'display dialog "Hyponoia is not installed yet. Double-click Install Hyponoia.command first." buttons {"OK"} default button "OK"'
  exit 1
fi

if ! ".venv/bin/python" -c 'import numpy, soundfile, librosa, pythonosc' >/dev/null 2>&1; then
  osascript -e 'display alert "Hyponoia needs repair" message "Run Install Hyponoia.command again. Your library and learning will be preserved."'
  exit 1
fi

exec ".venv/bin/python" "hyponoia_app.py"
