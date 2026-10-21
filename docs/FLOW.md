# Program Flow & Operational Guide

## Purpose

This document explains how the Airport X-Ray Queue Monitoring system works end-to-end, from camera input to dashboard display. It is intended for new developers joining the project and for presentation or demonstration purposes.

---

## 1. System Overview

The system monitors people queuing at an airport X-ray security checkpoint using a standard webcam. It detects and tracks people in real time, classifies queue congestion, and displays live analytics on a web dashboard.

**Key constraints:**

- Camera-only prototype — no physical sensors, no cloud services
- All processing runs locally on the host machine
- Detection uses YOLOv5s (COCO pretrained, class 0 = person)
- No facial recognition — track IDs are ephemeral and non-identifying

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER                                │
│                                                                      │
│   ┌──────────┐                                                       │
│   │  Camera   │  Webcam (index 0), video file, RTSP, or HTTP stream │
│   └─────┬────┘                                                       │
└─────────┼────────────────────────────────────────────────────────────┘
          │ frames
┌─────────▼────────────────────────────────────────────────────────────┐
│                     COMPUTER VISION PIPELINE                         │
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────────────────┐  │
│   │  YOLOv5s  │───▶│   SORT   │───▶│      QueueAnalytics          │  │
│   │ Detection │    │ Tracking │    │  (queue_prototype.py)        │  │
│   └──────────┘    └──────────┘    │                              │  │
│                                    │  Occupancy %                 │  │
│                                    │  Line crossings (ENTER/EXIT) │  │
│                                    │  Flow rates (1m, 5m)         │  │
│                                    │  Waiting time per person     │  │
│                                    │  Pixel spacing (density)     │  │
│                                    │  Rule-based classifier       │  │
│                                    │  Congestion stopwatch         │  │
│                                    └──────────────┬───────────────┘  │
└───────────────────────────────────────────────────┼──────────────────┘
                                                    │
┌───────────────────────────────────────────────────┼──────────────────┐
│                     SNAPSHOT LAYER                 │                   │
│                                                    │                  │
│   ┌────────────────────────────────────────────────▼───────────────┐ │
│   │              SnapshotBuilder (snapshot.py)                     │ │
│   │                                                                │ │
│   │  Converts internal analytics state into a JSON-serializable   │ │
│   │  QueueSnapshot object with all telemetry fields.              │ │
│   └───────────┬──────────────────────────────┬────────────────────┘ │
│               │                              │                      │
│   ┌───────────▼──────────┐    ┌──────────────▼──────────────────┐  │
│   │   SnapshotHolder     │    │     HistoryBuffer                │  │
│   │  (thread-safe, latest │    │  (ring buffer, 240 samples,     │  │
│   │   snapshot + frame)   │    │   4 min at 1 Hz)                │  │
│   └───────────┬──────────┘    └──────────────┬──────────────────┘  │
└───────────────┼──────────────────────────────┼──────────────────────┘
                │                              │
┌───────────────▼──────────────────────────────▼──────────────────────┐
│                      API LAYER (FastAPI)                             │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                api_server.py  (port 8080)                    │  │
│   │                                                              │  │
│   │  GET /api/dashboard/health    → camera + pipeline status    │  │
│   │  GET /api/dashboard/snapshot  → full dashboard telemetry    │  │
│   │  GET /api/dashboard/history   → 240-sample trend data       │  │
│   │  GET /api/dashboard/config    → camera ROI + layout         │  │
│   │  GET /api/camera/stream       → MJPEG live video            │  │
│   │  GET /api/snapshot            → raw internal snapshot       │  │
│   │  GET /api/health              → raw health check            │  │
│   └──────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────┼───────────────────────────────────┘
                                   │ HTTP (JSON + MJPEG)
┌──────────────────────────────────┼───────────────────────────────────┐
│                   DASHBOARD LAYER (React + Vite)                     │
│                                   │                                   │
│   ┌──────────────────────────────▼───────────────────────────────┐  │
│   │              Vite Dev Server (port 8443)                     │  │
│   │              Proxy: /api → localhost:8080                    │  │
│   └──────────────────────────────┬───────────────────────────────┘  │
│                                   │                                   │
│   ┌──────────────────────────────▼───────────────────────────────┐  │
│   │                    React Application                         │  │
│   │                                                              │  │
│   │  App.tsx                                                     │  │
│   │  ├── QueueStatusPanel    (state, occupancy, trend)          │  │
│   │  ├── CameraPanel         (MJPEG stream + tracking overlays) │  │
│   │  ├── KpiStack            (9 metric tiles with sparklines)   │  │
│   │  ├── AnalyticsPanel      (occupancy + flow charts)          │  │
│   │  ├── EventTimeline       (recent ENTER/EXIT events)         │  │
│   │  └── DiagnosticsDrawer   (raw snapshot, health details)     │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Stages in Detail

### Stage 1: Frame Capture

The system reads frames from the configured source (webcam, video file, or network stream) using OpenCV.

- **Source selection:** `config/camera.yaml` → `camera.source` (default: webcam index 0), overridden by `--source` CLI flag
- **Resolution:** Camera native resolution by default (`width: 0`, `height: 0`); configurable in `camera.yaml`
- **Frame rate:** Target 15 FPS (`TARGET_FPS = 15`); actual rate depends on camera and hardware

**Key file:** `app/detect_people.py` — `open_source()` function (line ~600)

### Stage 2: Person Detection (YOLO)

Each frame is processed by YOLOv5s to detect people.

- **Model:** YOLOv5s (COCO pretrained, 3.8 MB, class 0 = person)
- **Device:** Automatically selected — MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
- **Input size:** 320×320 pixels
- **Confidence threshold:** 0.25 (configurable via `--conf`)
- **Output:** Array of bounding boxes `[x1, y1, x2, y2, confidence]`

**Key file:** `app/detect_people.py` — `load_model()` and inference loop (line ~820)

### Stage 3: Multi-Object Tracking (SORT)

Detected people are tracked across frames using the SORT algorithm.

- **Algorithm:** Kalman filter + Hungarian assignment + IoU matching
- **Stable IDs:** Each person gets a persistent track ID that follows them through the frame
- **Track loss:** If a track is not matched for `track_lost_grace_seconds` (default 1.5s), it is dropped
- **Output:** Array of tracked boxes `[x1, y1, x2, y2, track_id]`

**Key file:** `app/sort_tracker.py`

### Stage 4: Queue Analytics

Track data is fed into the `QueueAnalytics` engine, which computes all queue metrics.

| Metric | Description | How It Works |
|--------|-------------|--------------|
| **Occupancy** | People in queue / capacity | Count centroids inside the ROI polygon |
| **Line crossings** | ENTER / EXIT events | Track centroids cross the counting line |
| **Flow rate** | People per minute | Rolling window (1-min and 5-min) |
| **Waiting time** | Entry→exit duration | Per-track stopwatch from ENTER to EXIT |
| **Spacing** | Average pixel distance | Pairwise Euclidean distance between centroids |
| **Movement** | MOVING / STAGNANT / IDLE | Line crossings within a rolling window |
| **State classification** | EMPTY → BACKLOG | Rule-based with hysteresis and debounce |

**Key file:** `app/queue_prototype.py` — `QueueAnalytics` class

### Stage 5: Snapshot Construction

The `SnapshotBuilder` converts internal analytics state into a JSON-serializable `QueueSnapshot` object.

- Combines queue metrics, system metrics, camera state, and events
- Events are assigned unique sequential IDs (via `enumerate()`)
- The snapshot is stored in the thread-safe `SnapshotHolder` for the API server

**Key files:** `app/snapshot.py`, `app/snapshot_holder.py`

### Stage 6: History Buffer

Each frame also appends a sample to the `HistoryBuffer` (ring buffer, 240 samples = ~4 minutes at 1 Hz).

- Contains: timestamp, queue count, occupancy %, entry/exit rates, average wait
- Thread-safe, fixed-capacity, oldest-first eviction
- Used by the `/api/dashboard/history` endpoint for trend charts

**Key file:** `app/snapshot_buffer.py`

### Stage 7: API Server (FastAPI)

The FastAPI server exposes HTTP endpoints for the React dashboard.

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/dashboard/health` | GET | Camera + pipeline status |
| `/api/dashboard/snapshot` | GET | Full dashboard telemetry (queue, system, camera, events, history, tracks) |
| `/api/dashboard/history` | GET | 240-sample trend data |
| `/api/dashboard/config` | GET | Camera ROI + layout configuration |
| `/api/camera/stream` | GET | MJPEG live video with tracking overlays |
| `/api/snapshot` | GET | Raw internal snapshot (legacy format) |
| `/api/health` | GET | Raw health check (legacy format) |

The server binds to `0.0.0.0:8080` by default (configurable via `--api-host` and `--api-port`).

**Key file:** `app/api_server.py`

### Stage 8: React Dashboard

The Vite dev server (port 8443) proxies `/api` requests to the FastAPI server (port 8080). The React application polls the API at ~1 Hz and renders the dashboard.

**Key file:** `Queue-monitoring-dashboard/src/App.tsx`

---

## 4. Key Files Reference

### Backend (`airport_people_counter/`)

| File | Purpose |
|------|---------|
| `app/detect_people.py` | Main entry point — CLI, camera loop, YOLO inference, SORT tracking, analytics, snapshot building |
| `app/queue_prototype.py` | `QueueAnalytics` engine — occupancy, line crossings, flow, waiting, spacing, classifier, stopwatch |
| `app/sort_tracker.py` | SORT multi-object tracker (Kalman + IoU) |
| `app/snapshot.py` | `SnapshotBuilder` — converts internal state to `QueueSnapshot` |
| `app/snapshot_holder.py` | Thread-safe holder for the latest snapshot + annotated frame |
| `app/snapshot_buffer.py` | Thread-safe ring buffer for 240-sample history |
| `app/dashboard_adapter.py` | Converts internal snapshot to dashboard-shaped `DashboardSnapshot` |
| `app/dashboard_models.py` | Pydantic/dataclass models for dashboard API responses |
| `app/api_server.py` | FastAPI server — HTTP endpoints for the React dashboard |
| `app/api_models.py` | Pydantic models for raw API responses |
| `config/camera.yaml` | All runtime configuration — camera, ROI, counting line, classifier thresholds |
| `run.sh` | Convenience launcher (auto-creates venv, installs deps) |

### Frontend (`Queue-monitoring-dashboard/`)

| File | Purpose |
|------|---------|
| `src/App.tsx` | Root component — layout, mode switching (LIVE/DEMO), data source selection |
| `src/components/sections.tsx` | Dashboard panels — `QueueStatusPanel`, `CameraPanel`, `KpiStack`, `AnalyticsPanel`, `EventTimeline` |
| `src/components/charts.tsx` | Chart components — `Sparkline`, `OccupancyChart`, `FlowChart` |
| `src/components/health.tsx` | `OperatorHealth` badge, `DiagnosticsDrawer` |
| `src/lib/api.ts` | API client — `getSnapshot()`, `getConfig()`, `getHealth()`, `getVideoStreamUrl()` |
| `src/lib/telemetry.ts` | Data hooks — `useRealTelemetry()` (live API), `useTelemetry()` (simulator) |
| `src/lib/ui.ts` | Shared UI primitives — `StatusChip`, `Panel`, formatters |
| `vite.config.ts` | Vite config — proxy `/api` → `localhost:8080`, port 8443 |

---

## 5. Configuration

All runtime configuration lives in `config/camera.yaml`. The most important fields:

```yaml
camera:
  source: 0                    # Webcam index, or video/RTSP URL
  width: 0                     # 0 = native resolution
  height: 0
  roi:                         # Queue area polygon (pixel coordinates)
    - [160, 100]
    - [1120, 100]
    - [1120, 520]
    - [160, 520]
  counting_line:                # Virtual line for ENTER/EXIT detection
    - [160, 560]
    - [1120, 560]
  xray_side: left              # Which side of the line is the X-ray machine
  queue_capacity: 30           # Max people → occupancy % calculation

classifier:
  occupancy_sparse: 0.10       # ≥10% → SPARSE
  occupancy_normal: 0.40       # ≥40% → NORMAL
  occupancy_crowded: 0.60      # ≥60% → CROWDED
  occupancy_congested: 0.80    # ≥80% → CONGESTED (requires spacing ≤ 60px)
  hysteresis: 0.05             # Anti-flapping buffer
  debounce_frames: 5           # Frames before state commits
```

Use `./run.sh --define-roi` to interactively draw the ROI and counting line with the webcam.

---

## 6. Data Flow: Backend to Dashboard

The backend sends a JSON payload from `/api/dashboard/snapshot` every ~1 second. Here is how each field maps to the React dashboard:

| Backend Field | Dashboard Component | Visual Element |
|---------------|---------------------|----------------|
| `queue.state` | `QueueStatusPanel` | State label (NORMAL, BUSY, CONGESTED, BACKLOG) |
| `queue.occupancy` | `QueueStatusPanel` | Occupancy percentage + progress bar |
| `queue.trend` | `QueueStatusPanel` | Trend arrow (GROWING, CLEARING, STEADY) |
| `queue.queue` | `KpiStack` | "People Waiting" tile |
| `queue.congestion_sec` | `QueueStatusPanel` | Congestion duration strip |
| `queue.avg_wait` | `KpiStack` | "Avg Wait" tile |
| `queue.entry_rate` | `KpiStack` + `FlowChart` | Entry rate tile + chart line |
| `queue.exit_rate` | `KpiStack` + `FlowChart` | Exit rate tile + chart line |
| `camera.link` | `QueueStatusPanel` | Camera status (ONLINE/OFFLINE/DEGRADED) |
| `camera.fps` | `CameraPanel` | FPS display |
| `camera.resolution` | `CameraPanel` | Resolution display (e.g., 1280x720) |
| `system.inference_ms` | `KpiStack` | Inference latency tile |
| `system.detections` | `KpiStack` | Detections count tile |
| `system.active_tracks` | `KpiStack` | Active tracks tile |
| `history[]` | `OccupancyChart` + `FlowChart` | Trend lines (240 samples) |
| `events[]` | `EventTimeline` | Recent ENTER/EXIT/state change events |
| `tracks[]` | `CameraPanel` | Bounding box overlays on MJPEG stream |

---

## 7. Dashboard Modes

### LIVE Mode (default)

- Polls `getSnapshot()` from the FastAPI backend at ~1 Hz
- Displays real camera data, real queue metrics, real events
- Requires the backend to be running: `./run.sh --mode headless --source 0 --api`

### DEMO Mode

- Uses `useTelemetry()` hook with synthetic data
- No backend required
- Selectable scenarios: NORMAL, BUSY, CONGESTED, BACKLOG, DEGRADED, OFFLINE
- Useful for UI development and presentation without a camera

---

## 8. Startup Commands

### Full Stack (Backend + Dashboard)

```bash
# Terminal 1: Start the YOLO pipeline + FastAPI server (headless, no OpenCV window)
cd airport_people_counter
./run.sh --mode headless --source 0 --api

# Terminal 2: Start the React dashboard
cd Queue-monitoring-dashboard
npm run dev
# Dashboard available at http://localhost:8443
```

### Standalone Backend (with OpenCV window)

```bash
cd airport_people_counter
./run.sh --source 0
# Opens a desktop window with live video + queue metrics overlay
```

### Headless Backend (API only, no GUI)

```bash
cd airport_people_counter
./run.sh --mode headless --source 0 --api
# API available at http://localhost:8080
```

### With Video File

```bash
cd airport_people_counter
./run.sh --source input/airport_cctv.mp4 --api --mode headless
```

### ROI Setup

```bash
cd airport_people_counter
./run.sh --define-roi
# Interactive: click to draw ROI polygon, counting line, X-ray side point
# Saves to config/camera.yaml
```

### Camera Diagnostics

```bash
cd airport_people_counter
./run.sh --list-cameras        # Find available camera indices
./run.sh --test-camera 0       # Test camera FPS without YOLO
./run.sh --roi-preview         # Preview saved ROI overlay
```

---

## 9. Verification Steps

After starting the full stack, verify everything works:

### Backend Health

```bash
curl -s http://localhost:8080/api/dashboard/health | python3 -m json.tool
# Expected: {"status":"ok","camera":true,"api":true,"last_frame_age_ms":<small number>}
```

### Snapshot Data

```bash
curl -s http://localhost:8080/api/dashboard/snapshot | python3 -m json.tool
# Expected: Full JSON with queue.state, queue.occupancy, camera.link:"ONLINE", events[], history[]
```

### MJPEG Stream

```bash
curl -N --max-time 3 http://localhost:8080/api/camera/stream > /dev/null
# Expected: Stream of JPEG frames (several MB received)
```

### Dashboard

1. Open `http://localhost:8443` in a browser
2. Confirm the LIVE toggle is selected (green dot, "REAL-TIME" label)
3. Verify: Camera panel shows live video with bounding boxes
4. Verify: Status panel shows queue state (e.g., NORMAL)
5. Verify: KPI tiles show non-zero values
6. Verify: Charts populate within ~4 seconds (240 samples at 1 Hz)
7. Verify: Console shows zero errors

---

## 10. Classifier State Machine

The rule-based classifier transitions through states based on occupancy ratio (people in queue / capacity):

```
EMPTY (occupancy < 10%)
  │ occupancy ≥ 10%
  ▼
SPARSE (occupancy < 40%)
  │ occupancy ≥ 40%
  ▼
NORMAL (occupancy < 60%)
  │ occupancy ≥ 60%
  ▼
CROWDED (occupancy < 80%)
  │ occupancy ≥ 80% AND spacing ≤ 60px
  ▼
CONGESTED
  │ all backlog conditions met for 10+ seconds
  ▼
BACKLOG (disabled by default — requires flow baseline)
```

**Hysteresis:** Downgrade threshold = threshold × (1 − 0.05). This prevents rapid state oscillation near boundaries.

**Debounce:** A candidate state must persist for 5 consecutive frames before committing.

---

## 11. Quick Reference

| Task | Command |
|------|---------|
| Start full stack | `./run.sh --mode headless --source 0 --api` |
| Start dashboard only | `cd Queue-monitoring-dashboard && npm run dev` |
| Check API health | `curl http://localhost:8080/api/dashboard/health` |
| Get live snapshot | `curl http://localhost:8080/api/dashboard/snapshot` |
| View MJPEG stream | `curl -N http://localhost:8080/api/camera/stream` |
| Run unit tests | `cd airport_people_counter && make test` |
| Run smoke test | `cd airport_people_counter && make smoke` |
| Define ROI | `./run.sh --define-roi` |
| Preview ROI | `./run.sh --roi-preview` |
| List cameras | `./run.sh --list-cameras` |
| Test camera FPS | `./run.sh --test-camera 0` |
| Record events | `./run.sh --record output/events.jsonl` |
| Record samples | `./run.sh --record-samples output/samples.jsonl` |
