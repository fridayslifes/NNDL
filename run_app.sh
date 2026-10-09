#!/usr/bin/env bash
# One-command launcher for macOS / Linux.
#   ./run_app.sh            -> sets everything up (first run only) and opens the dashboard
#   PORT=8010 ./run_app.sh  -> use another port
#
# What it does:
#   1. finds Python 3.10+                     2. creates a private virtual environment in .venv/
#   3. installs requirements.txt (first run)  4. starts the FastAPI app and opens your browser
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

# --- 1. find a suitable Python -------------------------------------------------
PYTHON=""
for candidate in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] else 1)' 2>/dev/null; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "ERROR: Python 3.10 or newer was not found."
  echo "Install it from https://www.python.org/downloads/ (or 'brew install python@3.12') and run this script again."
  exit 1
fi
echo "Using $($PYTHON --version) ($PYTHON)"

# --- 2. virtual environment ----------------------------------------------------
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv/ ..."
  "$PYTHON" -m venv .venv
fi

# --- 3. dependencies (re-installed only when requirements.txt changes) ---------
STAMP=".venv/.requirements-installed"
if [ ! -f "$STAMP" ] || ! cmp -s requirements.txt "$STAMP"; then
  echo "Installing dependencies (first run takes a few minutes: PyTorch is a large download) ..."
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
  cp requirements.txt "$STAMP"
fi

# --- 4. start the app ----------------------------------------------------------
for required in models/best_model.pth models/model_config.json models/scaler.pkl models/encoder.pkl; do
  if [ ! -f "$required" ]; then
    echo "WARNING: $required is missing — the dashboard will start but say 'Model not ready'."
    echo "         Run the notebooks 01-07 to create the model files."
  fi
done

URL="http://localhost:${PORT}"
echo
echo "Starting the dashboard at ${URL}   (press Ctrl+C to stop)"
if [ -z "${NO_BROWSER:-}" ]; then
  ( sleep 4; { command -v open >/dev/null 2>&1 && open "$URL"; } || { command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL"; } || true ) >/dev/null 2>&1 &
fi
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
