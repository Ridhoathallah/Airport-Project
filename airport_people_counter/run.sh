#!/usr/bin/env bash
# Convenience launcher for the Airport X-Ray Queue camera-only prototype.
#
# Runs the app with the project virtualenv so you don't need to activate it
# first (`source venv/bin/activate`). If the venv or its dependencies are
# missing, they are created/installed automatically.
#
# Usage:
#   ./run.sh                          # webcam
#   ./run.sh --define-roi             # set the queue ROI + counting line
#   ./run.sh --source input/video.mp4 # run on a file
#   ./run.sh --api                    # enable FastAPI server for dashboard
#   ./run.sh --help                   # all CLI options
set -euo pipefail
cd "$(dirname "$0")"

PY=venv/bin/python

if [ ! -x "$PY" ]; then
  echo "[run] virtualenv not found — creating venv/ ..." >&2
  if command -v python3.9 >/dev/null 2>&1; then
    python3.9 -m venv venv
  else
    python3 -m venv venv
  fi
fi

if ! "$PY" -c "import cv2, torch, numpy, yaml" >/dev/null 2>&1; then
  echo "[run] installing dependencies into venv/ ..." >&2
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements/requirements.txt
fi

export PYTHONUNBUFFERED=1
exec "$PY" app/detect_people.py "$@"
