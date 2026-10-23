#!/usr/bin/env bash
# ==============================================================================
# Airport X-Ray Queue Monitoring System — Project-level Unified Launcher
#
# One command starts the whole system:
#
#   camera -> YOLOv5 -> SORT -> QueueAnalytics -> FastAPI -> Vite dashboard
#
# It checks the environment, starts the YOLO pipeline in headless mode,
# waits for the API to answer, starts the dashboard, waits for it, and
# prints a status block. Ctrl+C stops both without leaving orphans.
#
# Usage:
#   ./start.sh                 # camera 0, dashboard on :8443
#   ./start.sh --source 1      # a different camera
#   ./start.sh --source clip.mp4
#   ./start.sh --no-dashboard  # backend only
#   ./start.sh --lan           # expose the dashboard on the LAN (default: localhost only)
#
# The browser never runs shell commands: it talks to the FastAPI service.
# Starting the system is a local developer/operator action, so it belongs
# in a launcher rather than in the dashboard.
# ==============================================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${ROOT_DIR}/airport_people_counter"
FRONTEND_DIR="${ROOT_DIR}/Queue-monitoring-dashboard"

API_HOST="127.0.0.1"
API_PORT="8080"
DASH_PORT="8443"
HEALTH_URL="http://${API_HOST}:${API_PORT}/api/dashboard/health"

SOURCE=""
START_DASHBOARD=1
DASH_HOST="127.0.0.1"
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --source)      SOURCE="${2:-}"; shift 2 ;;
    --source=*)    SOURCE="${1#*=}"; shift ;;
    --api-port)    API_PORT="${2:-}"; shift 2 ;;
    --api-port=*)  API_PORT="${1#*=}"; shift ;;
    --dash-port)   DASH_PORT="${2:-}"; shift 2 ;;
    --dash-port=*) DASH_PORT="${1#*=}"; shift ;;
    --no-dashboard) START_DASHBOARD=0; shift ;;
    --lan)         DASH_HOST="0.0.0.0"; shift ;;
    -h|--help)
      sed -n '3,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)             PASSTHROUGH+=("$1"); shift ;;
  esac
done

BACKEND_PID=""
FRONTEND_PID=""
LOG_DIR="${ROOT_DIR}/.run"
mkdir -p "${LOG_DIR}"

# Bound the logs. The pipeline writes a status line every second; a long
# run would otherwise grow the file without limit. Anything past the cap
# is rotated to a single .1 file, so the directory never exceeds 2x the cap.
LOG_MAX_BYTES=$((8 * 1024 * 1024))
rotate_log() {
  local file="$1"
  [ -f "${file}" ] || return 0
  local size
  size=$(wc -c < "${file}" 2>/dev/null || echo 0)
  if [ "${size}" -gt "${LOG_MAX_BYTES}" ]; then
    mv -f "${file}" "${file}.1"
    : > "${file}"
  fi
}
rotate_log "${LOG_DIR}/backend.log"
rotate_log "${LOG_DIR}/frontend.log"

CLEANED=0
cleanup() {
  [ "${CLEANED}" = "1" ] && return 0
  CLEANED=1
  echo ""
  echo "Shutting down Airport X-Ray Queue Monitoring System..."
  # The pipeline stops itself when it receives the shutdown request; give it
  # a moment, then make sure it is gone either way.
  if [ -n "${BACKEND_PID}" ] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    curl -s -m 3 -X POST "http://${API_HOST}:${API_PORT}/api/shutdown" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "${BACKEND_PID}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${BACKEND_PID}" 2>/dev/null; then
      kill -TERM "${BACKEND_PID}" 2>/dev/null || true
      sleep 1
    fi
    pkill -TERM -P "${BACKEND_PID}" 2>/dev/null || true
    kill -KILL "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
  if [ -n "${FRONTEND_PID}" ] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    pkill -TERM -P "${FRONTEND_PID}" 2>/dev/null || true
    kill -TERM "${FRONTEND_PID}" 2>/dev/null || true
    wait "${FRONTEND_PID}" 2>/dev/null || true
  fi
  echo "Clean shutdown complete."
  exit 0
}
trap cleanup INT TERM EXIT

fail() {
  echo "" >&2
  echo "[start] FAILED: $1" >&2
  echo "[start] backend log : ${LOG_DIR}/backend.log" >&2
  [ "${START_DASHBOARD}" = "1" ] && echo "[start] frontend log: ${LOG_DIR}/frontend.log" >&2
  exit 1
}

# ── 1. Environment ─────────────────────────────────────────────────────────
PY="${BACKEND_DIR}/venv/bin/python"
if [ ! -x "${PY}" ]; then
  echo "[start] Python virtualenv not found in ${BACKEND_DIR}/venv" >&2
  echo "[start] Creating it now (this takes a minute)..." >&2
  (cd "${BACKEND_DIR}" && python3 -m venv venv) || fail "could not create the virtualenv"
  PY="${BACKEND_DIR}/venv/bin/python"
fi
if ! "${PY}" -c "import cv2, torch, numpy, yaml" >/dev/null 2>&1; then
  echo "[start] Backend dependencies missing, installing..." >&2
  "${PY}" -m pip install -q -r "${BACKEND_DIR}/requirements/requirements.txt" \
    || fail "dependency installation failed"
fi
echo "[start] Python environment ready."

# ── 2. Reuse a running API instead of starting a second one ────────────────
YOLO_STATUS="READY"
CAMERA_STATUS="UNKNOWN"
FASTAPI_STATUS="UNKNOWN"

if curl -s -f -m 2 "${HEALTH_URL}" >/dev/null 2>&1; then
  FASTAPI_STATUS="READY"
  echo "[start] Reusing the API already listening on ${API_PORT}."
else
  BACKEND_ARGS=(--mode headless --api --api-port "${API_PORT}")
  [ -n "${SOURCE}" ] && BACKEND_ARGS+=(--source "${SOURCE}")
  [ ${#PASSTHROUGH[@]} -gt 0 ] && BACKEND_ARGS+=("${PASSTHROUGH[@]}")

  echo "[start] Starting the YOLO pipeline (headless)..."
  ( cd "${BACKEND_DIR}" && exec "${PY}" app/detect_people.py "${BACKEND_ARGS[@]}" ) \
    >"${LOG_DIR}/backend.log" 2>&1 &
  BACKEND_PID=$!

  # ── 3. Wait for the API to answer ───────────────────────────────────────
  READY=0
  for _ in $(seq 1 60); do
    if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
      fail "the YOLO pipeline exited during startup (see ${LOG_DIR}/backend.log)"
    fi
    if curl -s -f -m 2 "${HEALTH_URL}" >/dev/null 2>&1; then READY=1; break; fi
    sleep 1
  done
  [ "${READY}" = "1" ] || fail "the API did not answer on port ${API_PORT} within 60s"
  FASTAPI_STATUS="READY"
fi

# ── 4. Report the camera honestly ─────────────────────────────────────────
# The health endpoint knows whether frames are arriving. Reporting READY
# regardless would be a lie the operator has to debug later.
# The API answers before the first frame has been processed, so a single
# check would report "no signal" for a camera that is working. Give the
# pipeline time to deliver frames, then report what is actually true.
CAMERA_STATUS="NO SIGNAL"
for _ in $(seq 1 20); do
  CAM_CHECK="$(curl -s -m 3 "${HEALTH_URL}" 2>/dev/null || echo '{}')"
  if echo "${CAM_CHECK}" | grep -q '"camera":true'; then
    CAMERA_STATUS="READY"
    break
  fi
  sleep 1
done

# ── 5. Dashboard ───────────────────────────────────────────────────────────
DASHBOARD_STATUS="SKIPPED"
if [ "${START_DASHBOARD}" = "1" ]; then
  if curl -s -f -m 2 "http://127.0.0.1:${DASH_PORT}" >/dev/null 2>&1; then
    DASHBOARD_STATUS="READY"
    echo "[start] Reusing the dashboard already listening on ${DASH_PORT}."
  else
    if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
      echo "[start] Installing dashboard dependencies..."
      ( cd "${FRONTEND_DIR}" && npm install --no-audit --no-fund ) >"${LOG_DIR}/npm.log" 2>&1 \
        || fail "npm install failed (see ${LOG_DIR}/npm.log)"
    fi
    echo "[start] Starting the dashboard..."
    VITE_BIN="${FRONTEND_DIR}/node_modules/.bin/vite"
    if [ ! -x "${VITE_BIN}" ]; then
      fail "the dashboard dependencies are missing (run npm install in ${FRONTEND_DIR})"
    fi
    # Run vite directly: going through "npm run dev" would leave the real
    # vite process as a grandchild, and signalling the npm wrapper would
    # orphan it.
    ( cd "${FRONTEND_DIR}" && exec "${VITE_BIN}" --port "${DASH_PORT}" --host "${DASH_HOST}" ) \
      >"${LOG_DIR}/frontend.log" 2>&1 &
    FRONTEND_PID=$!

    READY=0
    for _ in $(seq 1 40); do
      if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        fail "the dashboard exited during startup (see ${LOG_DIR}/frontend.log)"
      fi
      if curl -s -f -m 2 "http://127.0.0.1:${DASH_PORT}" >/dev/null 2>&1; then READY=1; break; fi
      sleep 1
    done
    [ "${READY}" = "1" ] || fail "the dashboard did not answer on port ${DASH_PORT} within 40s"
    DASHBOARD_STATUS="READY"
  fi
fi

# ── 6. Status block ───────────────────────────────────────────────────────
echo ""
echo "Airport X-Ray Queue Monitoring"
echo "--------------------------------"
printf "%-12s%s\n" "YOLO" "${YOLO_STATUS}"
printf "%-12s%s\n" "Camera" "${CAMERA_STATUS}"
printf "%-12s%s\n" "FastAPI" "${FASTAPI_STATUS}"
printf "%-12s%s\n" "Dashboard" "${DASHBOARD_STATUS}"
echo ""
echo "Open:"
if [ "${START_DASHBOARD}" = "1" ]; then
  echo "http://localhost:${DASH_PORT}"
else
  echo "http://localhost:${API_PORT}/docs"
fi
echo ""
echo "Press Ctrl+C to stop."
echo ""

# ── 7. Stay in the foreground, and do not leave orphans ───────────────────
while true; do
  if [ -n "${BACKEND_PID}" ] && ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[start] the YOLO pipeline stopped."
    break
  fi
  if [ -n "${FRONTEND_PID}" ] && ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "[start] the dashboard stopped."
    break
  fi
  sleep 1 &
  wait $! 2>/dev/null || true
done
cleanup
