# Airport X-Ray Queue Monitoring System

> Airport security checkpoint queue analytics using real-time computer vision.
> Detects, tracks, and classifies people in queue from a standard webcam feed.
> All processing runs locally — no cloud services, no facial recognition.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    CAMERA / VIDEO INPUT                      │
│  Webcam (index 0) · Video file · RTSP/HTTP stream           │
└───────────────────────────┬─────────────────────────────────┘
                            │ frames
┌───────────────────────────▼─────────────────────────────────┐
│                  COMPUTER VISION PIPELINE                    │
│                                                              │
│  YOLOv5s Detection ──▶ SORT Tracking ──▶ QueueAnalytics     │
│  (COCO person class)   (Kalman + IoU)   (ROI, flow, state)  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    SNAPSHOT LAYER                             │
│  SnapshotBuilder → SnapshotHolder → HistoryBuffer            │
│  (JSON-serializable)   (thread-safe)    (240 samples, 4min) │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     API LAYER (FastAPI)                       │
│  http://localhost:8080                                        │
│  /api/dashboard/snapshot · /api/dashboard/health             │
│  /api/dashboard/history · /api/camera/stream (MJPEG)         │
│  /api/shutdown                                               │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP proxy
┌───────────────────────────▼─────────────────────────────────┐
│                 REACT DASHBOARD (Vite)                        │
│  http://localhost:8443                                        │
│  Queue Status · Camera Feed · KPIs · Charts · Events         │
│  LIVE mode (real API) · DEMO mode (simulated data)           │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle:** The Python backend is the **source of truth** for all queue analytics. The React dashboard is a **presentation layer only** — it receives data via API and renders it.

---

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9 | Apple Silicon (MPS) or x86 (CPU) |
| Node.js | 18+ | For dashboard |
| Camera | Any | Webcam, video file, or RTSP stream |

### Option 1 — Unified Launcher (Recommended)

```bash
cd Airport-Project
./start.sh
```

This starts both the backend (YOLO + FastAPI on port 8080) and the dashboard (Vite on port 8443). Press `Ctrl+C` to stop both.

### Option 2 — Manual (Two Terminals)

**Terminal 1 — Backend:**
```bash
cd airport_people_counter
./run.sh --headless --api --source 0
```

**Terminal 2 — Dashboard:**
```bash
cd Queue-monitoring-dashboard
npm run dev
```

### Open Dashboard

Open **http://localhost:8443** in your browser.

- The dashboard starts in **LIVE** mode by default
- Select **DEMO** mode for simulated data (no camera needed)
- Click **"Shut Down"** in the header to stop the backend

---

## What It Does

The system monitors people queuing at an airport X-ray security checkpoint:

| Feature | Description |
|---------|-------------|
| **Person Detection** | YOLOv5s (COCO class 0) — 480px input, ~27ms inference on MPS |
| **Multi-Object Tracking** | SORT algorithm — Kalman filter + Hungarian assignment + IoU |
| **Queue ROI** | Configurable polygon defining the queue area |
| **Line Counting** | ENTER/EXIT events on separate IN/OUT lines with debounce |
| **Occupancy** | People in queue / capacity percentage |
| **Flow Rate** | Entry and exit rates (1-min and 5-min rolling windows) |
| **Waiting Time** | Per-person stopwatch from entry to exit |
| **State Classification** | Rule-based: EMPTY → SPARSE → NORMAL → CROWDED → CONGESTED |
| **Movement Detection** | MOVING / STAGNANT / IDLE based on line crossings |
| **Live Dashboard** | Real-time web UI with charts, KPIs, and event timeline |

---

## Configuration

All runtime configuration lives in `airport_people_counter/config/camera.yaml`:

```yaml
camera:
  source: 0                    # Webcam index, or video/RTSP URL
  width: 1280                  # Capture width (0 = native)
  height: 720                  # Capture height (0 = native)
  roi:                         # Queue area polygon (pixel coordinates)
    - [160.0, 100.0]
    - [1120.0, 100.0]
    - [1120.0, 520.0]
    - [160.0, 520.0]
  counting_line:                # Virtual line for ENTER/EXIT detection
    - [160.0, 560.0]
    - [1120.0, 560.0]
  xray_side_point: [640.0, 650.0]  # Reference point on X-ray side
  queue_capacity: 30           # Max people for occupancy % calculation

classifier:
  occupancy_sparse: 0.10       # ≥10% → SPARSE
  occupancy_normal: 0.40       # ≥40% → NORMAL
  occupancy_crowded: 0.60      # ≥60% → CROWDED
  occupancy_congested: 0.80    # ≥80% → CONGESTED
  hysteresis: 0.05             # Anti-flapping buffer
  debounce_frames: 5           # Frames before state commits
```

**Interactive ROI setup:**
```bash
cd airport_people_counter
./run.sh --define-roi
# Click to draw ROI polygon, counting line, and X-ray side point
# Saves to config/camera.yaml
```

**Interactive mapping & calibration (preferred):** everything above can be
calibrated from the dashboard itself — open the page and press
**"Edit mapping"** on the Live camera panel. Queue ROI and the IN/OUT lines
are drawn on the raw camera feed and stored per camera (normalised, so a
resolution change does not invalidate them) in
`airport_people_counter/config/mappings.yaml`. The detector hot-reloads the
geometry on the next frame; `camera.yaml` geometry remains as a legacy
fallback for deployments without a mapping.

---

## CLI Reference

### Backend (`run.sh`)

```bash
./run.sh                              # Webcam, standalone (OpenCV window)
./run.sh --headless --api             # Headless + FastAPI (for dashboard)
./run.sh --source input/video.mp4     # Run on video file
./run.sh --define-roi                 # Interactive ROI setup
./run.sh --roi-preview                # Preview saved ROI overlay
./run.sh --list-cameras               # Find available camera indices
./run.sh --test-camera 0              # Test camera FPS without YOLO
./run.sh --api --api-port 8080        # Custom API port
./run.sh --conf 0.3                   # Detection confidence threshold
./run.sh --record output/events.jsonl # Record events to JSONL
./run.sh --help                       # All options
```

### Unified Launcher (`start.sh`)

```bash
./start.sh                    # Start both backend + dashboard
./start.sh --source 1         # Use a different camera
./start.sh --source clip.mp4  # Use a video file
./start.sh --no-dashboard     # Backend only
Ctrl+C                        # Graceful shutdown of both services
```

The launcher checks the environment, starts the YOLO pipeline headless,
waits for the API, starts the dashboard, waits for it, and prints a status
block. It reuses an API or dashboard that is already running rather than
starting a second copy, and it reports the camera as `NO SIGNAL` when
frames are not arriving.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/snapshot` | GET | Full dashboard telemetry (queue, camera, KPIs, events, history, tracks) |
| `/api/dashboard/health` | GET | Camera + pipeline health status |
| `/api/dashboard/history` | GET | 240-sample trend data (4 minutes at 1 Hz) |
| `/api/dashboard/config` | GET | Camera ROI + layout configuration (+ additive mapping fields) |
| `/api/dashboard/mapping` | GET/PUT | Per-camera queue ROI + IN/OUT lines (mapping editor) |
| `/api/camera/stream` | GET | MJPEG live video with tracking overlays (`?annotated=0` = raw, for calibration) |
| `/api/snapshot` | GET | Raw internal snapshot (legacy format) |
| `/api/health` | GET | Raw health check (legacy format) |
| `/api/shutdown` | POST | Gracefully shutdown the backend |

**Quick check:**
```bash
curl -s http://localhost:8080/api/dashboard/health | python3 -m json.tool
curl -s http://localhost:8080/api/dashboard/snapshot | python3 -m json.tool
```

---

## Project Structure

```
Airport-Project/
├── airport_people_counter/          # Python backend (YOLO + FastAPI)
│   ├── app/
│   │   ├── detect_people.py         # Main entry — camera loop, YOLO, SORT, analytics
│   │   ├── queue_prototype.py       # QueueAnalytics engine
│   │   ├── sort_tracker.py          # SORT multi-object tracker
│   │   ├── snapshot.py              # SnapshotBuilder — internal → JSON
│   │   ├── snapshot_holder.py       # Thread-safe snapshot store
│   │   ├── snapshot_buffer.py       # History ring buffer (240 samples)
│   │   ├── dashboard_adapter.py     # Internal snapshot → dashboard format
│   │   ├── dashboard_models.py      # Pydantic models for dashboard API
│   │   ├── api_server.py            # FastAPI server (port 8080)
│   │   ├── api_models.py            # Pydantic models for raw API
│   │   └── shutdown.py              # Cooperative shutdown (loop + server)
│   ├── ml/                          # ML dataset & experiment tooling
│   │   ├── contract.py              # Data contract, ids, hashes, class schema
│   │   ├── manifest.py              # Versioned, hash-verified dataset manifests
│   │   ├── validate.py              # Annotation + dataset validation
│   │   ├── split.py                 # Deterministic group-based splitting
│   │   ├── catalog.py               # SQLite metadata catalog
│   │   ├── preflight.py             # Training readiness checks
│   │   ├── experiment.py            # Experiment provenance
│   │   ├── evaluate.py              # Evaluation reports and gates
│   │   └── collect.py               # Hard-example mining
│   ├── scripts/                     # CLI tools (evaluation + dataset tooling)
│   ├── tests/                       # 294 tests (pytest + unittest)
│   ├── datasets/                    # Dataset versions, manifests, catalog
│   ├── config/camera.yaml           # Runtime configuration
│   ├── models/yolov5s.pt            # YOLOv5s weights (auto-downloaded)
│   ├── requirements/                # Python dependencies
│   └── run.sh                       # Backend launcher
├── Queue-monitoring-dashboard/      # React + Vite frontend
│   ├── src/
│   │   ├── App.tsx                  # Root component — layout, mode switching
│   │   ├── components/
│   │   │   ├── sections.tsx         # Dashboard panels (status, camera, KPIs, events)
│   │   │   ├── charts.tsx           # Sparkline, OccupancyChart, FlowChart
│   │   │   └── health.tsx           # OperatorHealth badge, DiagnosticsDrawer
│   │   └── lib/
│   │       ├── api.ts               # API client functions
│   │       ├── telemetry.ts         # Data hooks (live + simulator)
│   │       └── ui.ts                # Shared UI primitives
│   ├── vite.config.ts               # Proxy /api → localhost:8080
│   └── package.json
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # System architecture deep-dive
│   ├── OPERATIONS.md                # Install, run, configure, troubleshoot
│   ├── DEVELOPMENT.md               # Dev workflow, testing, debugging
│   ├── FLOW.md                      # End-to-end pipeline walkthrough
│   ├── CHANGES.md                   # Integration bug fixes log
│   ├── API.md                       # Endpoint reference with schemas
│   ├── DATASETS.md                 # Dataset engineering, fine-tuning workflow
│   ├── CONFIGURATION.md             # camera.yaml schema reference
│   ├── ROADMAP.md                   # Implementation status + future plans
│   └── VALIDATION.md                # Test methodology and scenarios
├── start.sh                         # Unified launcher (both services)
└── README.md                        # This file
```

---

## Classifier State Machine

The rule-based classifier transitions through states based on queue occupancy:

```
EMPTY (occupancy < 10%)
  │
  ▼  occupancy ≥ 10%
SPARSE (occupancy < 40%)
  │
  ▼  occupancy ≥ 40%
NORMAL (occupancy < 60%)
  │
  ▼  occupancy ≥ 60%
CROWDED (occupancy < 80%)
  │
  ▼  occupancy ≥ 80% AND spacing ≤ 60px
CONGESTED
  │
  ▼  sustained for 10+ seconds
BACKLOG (disabled — requires flow baseline)
```

- **Hysteresis:** Downgrade threshold = threshold × 0.95 (prevents rapid oscillation)
- **Debounce:** Candidate state must persist 5 frames before committing

---

## Verification

After starting the system, verify everything works:

| Check | Command / Action | Expected |
|-------|-----------------|----------|
| Backend health | `curl -s localhost:8080/api/dashboard/health` | `{"status":"OK","camera":true}` |
| Snapshot data | `curl -s localhost:8080/api/dashboard/snapshot` | Full JSON with queue, camera, events |
| MJPEG stream | `curl -N --max-time 3 localhost:8080/api/camera/stream` | Several MB of JPEG frames |
| Dashboard UI | Open http://localhost:8443 | Live video, status, KPIs, charts |
| Console errors | Check browser console | Zero errors (favicon 404 is OK) |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, data flow, component responsibilities |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Install, run, configure, camera setup, troubleshooting |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Dev workflow, testing, debugging, regression testing |
| [docs/FLOW.md](docs/FLOW.md) | End-to-end pipeline walkthrough with diagrams |
| [docs/CHANGES.md](docs/CHANGES.md) | Integration bug fixes and improvements log |
| [docs/API.md](docs/API.md) | FastAPI endpoint reference with request/response schemas |
| [docs/DATASETS.md](docs/DATASETS.md) | Dataset engineering, validation, fine-tuning and evaluation workflow |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | camera.yaml schema with IMPLEMENTED/PLANNED markers |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Current implementation vs future roadmap |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Validation methodology, test scenarios, sync checks |

---

## Explicitly NOT Implemented

These features are **not part of this prototype** and are listed to set clear boundaries:

- Physical sensors (HC-SR04, infrared, ToF, LiDAR)
- MQTT / IoT gateway / sensor fusion
- Cloud processing / production deployment
- Facial recognition / passenger identification
- Physical density measurement (people/m² — requires calibration)
- Backlog classification (disabled — needs flow baseline)

---

## License

Internal project — not published.
