#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"
LOG_FILE="$SCRIPT_DIR/install.log"

show_error() {
  /usr/bin/osascript -e 'display alert "Hyponoia was not installed" message "Open install.log in the Hyponoia folder for details. Nothing in your sound library was changed."'
}

trap show_error ERR
: > "$LOG_FILE"

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display alert "Python 3 is required" message "Install Python 3.10, 3.11 or 3.12, then run Install Hyponoia again."'
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)'; then
  osascript -e 'display alert "Unsupported Python version" message "Hyponoia currently supports Python 3.10, 3.11 or 3.12."'
  exit 1
fi

echo "Creating isolated Hyponoia environment…" | tee -a "$LOG_FILE"
python3 -m venv .venv >> "$LOG_FILE" 2>&1
".venv/bin/python" -m pip install --upgrade pip >> "$LOG_FILE" 2>&1
".venv/bin/python" -m pip install -r requirements-app.txt >> "$LOG_FILE" 2>&1

VOICE_MESSAGE="The local voice model will download automatically the first time you use voice feedback."
if ".venv/bin/python" install_voice_model_v1.py >> "$LOG_FILE" 2>&1; then
  VOICE_MESSAGE="The local Greek/English voice model is ready."
fi

LANGUAGE_MESSAGE="In Teach Hyponoia, press 'Enable context understanding' once. Hyponoia will guide you through the private local-language setup."

trap - ERR
VOICE_MESSAGE="$VOICE_MESSAGE" LANGUAGE_MESSAGE="$LANGUAGE_MESSAGE" /usr/bin/osascript <<'APPLESCRIPT'
display dialog "Hyponoia is installed. " & (system attribute "VOICE_MESSAGE") & return & return & (system attribute "LANGUAGE_MESSAGE") & return & return & "Double-click Open Hyponoia.command to begin." buttons {"OK"} default button "OK"
APPLESCRIPT
