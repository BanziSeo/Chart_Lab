#!/usr/bin/env bash
# Chart Trainer launcher for Unix-like systems
# Mirrors run_app.bat: creates venv, installs deps and starts Streamlit app

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Locate Python 3 interpreter
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[ERROR] Python 3.x is required but not found in PATH." >&2
    exit 1
fi

# Ensure Python version is 3.x
$PY - <<'PY'
import sys
if sys.version_info.major < 3:
    raise SystemExit(1)
PY

# Create virtual environment if needed and install deps
if [ ! -d venv ]; then
    "$PY" -m venv venv
    . venv/bin/activate
    "$PY" -m pip install --upgrade pip
    pip install -r "$ROOT/Chart_Lab/requirements.txt"
else
    . venv/bin/activate
fi

# Launch Streamlit app
"$PY" -m streamlit run "$ROOT/Chart_Lab/app.py"
