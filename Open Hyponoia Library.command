#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  osascript -e 'display dialog "Hyponoia is not installed yet. Please follow INSTALL_MAC.md first." buttons {"OK"} default button "OK"'
  exit 1
fi

exec ".venv/bin/python" "hyponoia_library_app.py"
