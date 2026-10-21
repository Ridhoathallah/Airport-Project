# Architecture Documentation

## System Overview

The Airport X-Ray Queue Monitoring System is a **camera-only prototype** that transforms a live camera feed into queue analytics for airport X-ray screening checkpoints. The system runs entirely on a single machine (macOS/Linux) with no cloud dependencies.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

CAMERA / VIDEO FILE / IMAGE / RTSP STREAM
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FRAMESTREAM (background thread)                                            │
│  • Opens source (cv2.VideoCapture)                                          │
│  • Requests 30 FPS (CAP_PROP_FPS)                                           │
│  • Paces to target FPS (--fps, default 30)                                  │
│  • Delivers frames to main loop                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  YOLOv5s PERSON DETECTION (Vision Layer)                                    │
│  • app/detect_people.py: load_model(), infer()                              │
│  • Model: models/yolov5s.pt (COCO class 0 = person)                         │
│  • Device: MPS (Apple Silicon) / CPU / CUDA (--device)                      │
│  • Inference size: --imgsz (default 480px)                                  │
│  • Output: detections[N, 5] = (x1, y1, x2, y2, confidence)                  │
│  • Detection modes:                                                         │
│      - "person" (default): full body boxes → SORT                           │
│      - "head": top --head-ratio fraction → SORT (steep angles)              │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SORT TRACKING (app/sort_tracker.py)                                        │
│  • Kalman filter per track (constant velocity)                              │
│  • IoU-based association + Hungarian algorithm (scipy.optimize)             │
│  • Parameters: max_age=30, min_hits=3, iou_threshold=0.3                    │
│  • Input: detections[N, 5]                                                  │
│  • Output: tracks[M, 5] = (x1, y1, x2, y2, track_id)                        │
│  • Stable IDs across frames; handles occlusion via prediction               │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  QUEUE ANALYTICS (app/queue_prototype.py)                                   │
│                                                                             │
│  QueueAnalytics.update(tracks, frame_shape, track_confs)                    │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │ ROI CHECK        │  │ COUNTING LINE    │  │ WAITING TIME             │  │
│  │ • centroid in    │  │ • Segment crosses│  │ • Timer starts on        │  │
│  │   polygon?       │  │   infinite line  │  │   ROI entry (or 1st      │  │
│  │ • people_in_queue│  │ • Direction via  │  │   line crossing)         │  │
│  │   = count        │  │   xray_side_point│  │ • Ends on EXIT crossing  │  │
│  └──────────────────┘  │ • ENTER: away    │  │ • Per-track total_wait_s │  │
│                        │   EXIT: toward   │  └──────────────────────────┘  │
│                        └──────────────────┘  ┌──────────────────────────┐  │
│  ┌──────────────────┐  ┌──────────────────┐  │ PIXEL SPACING            │  │
│  │ FLOW RATES       │  │ OCCUPANCY %      │  │ • Pairwise centroid      │  │
│  │ • 1-min & 5-min  │  │ people_in_queue /│  │   distances (pixels)     │  │
│  │   rolling windows│  │   queue_capacity │  │ • avg/min/median/px      │  │
│  │ • ENTER rate     │  │   × 100          │  └──────────────────────────┘  │
│  │ • EXIT rate      │  └──────────────────┘                                │
│  └──────────────────┘                                                    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ RULECLASSIFIER (transparent, configurable)                           │  │
│  │ States: EMPTY → SPARSE → NORMAL → CROWDED → CONGESTED → BACKLOG     │  │
│  │ • Occupancy ladder with hysteresis (0.05) + debounce (5 frames)     │  │
│  │ • CONGESTED requires mean spacing ≤ congested_max_spacing_px (60)   │  │
│  │ • BACKLOG: sustained multi-condition (count, occupancy, flow, wait) │  │
│  │ • Every snapshot carries exact reasons[] for its state              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CONGESTION STOPWATCH                                                  │  │
│  │ • Starts on CONGESTED/BACKLOG entry → congestion_start event         │  │
│  │ • Accumulates duration_s while severe                                 │  │
│  │ • Stops on exit from severe → congestion_end event                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ EVENT LOG                                                             │  │
│  │ track_enter_roi, line_crossing (ENTER/EXIT), track_lost,            │  │
│  │ state_change, congestion_start, congestion_end                       │  │
│  │ JSONL output via --record                                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD MODEL (build_dashboard_model)                                    │
│  • QueueMetrics: state, occupancy, flow, waiting, movement, spacing...     │
│  • SystemMetrics: fps, inference_ms, tracking_ms, analytics_ms, detections │
│  • Single source of truth for both console line and OpenCV dashboard       │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SNAPSHOT BUILDER (app/snapshot.py)                                         │
│  SnapshotBuilder.build(dashboard, system, config, tracks, events, frame)   │
│                                                                             │
│  Input:  DashboardModel + SystemMetrics + CameraConfig + tracks[] + events │
│  Output: QueueSnapshot (frozen dataclass) + annotated_frame                │
│                                                                             │
│  QueueSnapshot fields:                                                      │
│  • timestamp, frame_id, freshness (LIVE/STALE/OFFLINE), staleness_ms       │
│  • fps, inference_ms, tracking_ms, analytics_ms                            │
│  • detection_count, tracked_people, people_in_queue                        │
│  • total_in, total_out, queue_state, occupancy_percent                     │
│  • entry_rate, exit_rate, avg_wait_s, longest_wait_s                       │
│  • congestion_duration_s, spacing_px, movement                             │
│  • config_status, calibration, camera_state                                │
│  • tracks[] (normalized 0..1), events[]                                    │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ├──► draw_dashboard() / draw_overlay() → OpenCV Window
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SNAPSHOT HOLDER (app/snapshot_holder.py)                                   │
│  • Thread-safe singleton: SnapshotHolder                                    │
│  • Writer: camera engine (each frame) — update(snapshot, frame)             │
│  • Reader: FastAPI endpoints — get_latest_snapshot(), get_latest_frame()   │
│  • Frame copied for MJPEG encoding                                          │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FASTAPI SERVER (app/api_server.py) — Port 8080                             │
│  Endpoints:                                                                 │
│  • GET /api/snapshot      → QueueSnapshot (1 Hz poll target)               │
│  • GET /api/health        → Camera/CV health status                        │
│  • GET /api/config        → Camera configuration (read-only)               │
│  • GET /api/events        → Recent events (limit 50)                       │
│  • GET /api/camera/stream → MJPEG multipart/x-mixed-replace                │
│                                                                             │
│  CORS: http://localhost:8443, http://127.0.0.1:8443                        │
│  Conversion: internal QueueSnapshot → Pydantic API models                  │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VITE DEV SERVER (Queue-monitoring-dashboard) — Port 8443                   │
│  • /api/* proxied to http://localhost:8080                                 │
│  • React 19 + TypeScript + Tailwind CSS v4                                 │
│  • Hot reload on source changes                                            │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  REACT DASHBOARD (src/)                                                     │
│                                                                             │
│  App.tsx:                                                                   │
│  • Source selector: "simulator" | "api"                                    │
│  • Scenario selector (simulator): NORMAL/BUSY/CONGESTED/BACKLOG/OFFLINE    │
│  • Composes: QueueStatusPanel, CameraPanel, KpiStack, AnalyticsPanel,      │
│    EventTimeline, DiagnosticsDrawer                                        │
│                                                                             │
│  lib/telemetry.ts:                                                          │
│  • useTelemetry(scenario) — Simulator (synthetic queue dynamics)           │
│  • useRealTelemetry() — Polls /api/snapshot @ 1 Hz, converts → Snapshot    │
│                                                                             │
│  lib/api.ts:                                                                │
│  • getSnapshot(), getConfig(), getHealth(), getVideoStreamUrl()            │
│  • Type definitions matching Pydantic models                               │
│                                                                             │
│  components/sections.tsx:                                                   │
│  • CameraPanel — MJPEG <img> (API) or simulated CCTV scene (simulator)     │
│    Overlays: tracks, ROI, counting line, IDs (toggleable)                  │
│  • QueueStatusPanel — Main status chip, occupancy meter, trend/wait strip  │
│  • KpiStack — 9 metric tiles with sparklines                               │
│  • AnalyticsPanel — FlowChart + OccupancyChart with range selector         │
│  • EventTimeline — Filterable (All/Flow/Alerts) event log                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | Input | Output | Must NOT |
|-----------|---------------|-------|--------|----------|
| `FrameStream` | Camera capture + pacing | Source (index/path/URL) | Frames (BGR numpy) | Process frames |
| `YOLOv5s` | Person detection | Frame | detections[N,5] | Track, classify |
| `SortTracker` | Multi-object tracking | detections[N,5] | tracks[M,5] | Detect, classify |
| `QueueAnalytics` | All queue metrics | tracks[], frame_shape, confs | Snapshot dict + events | Render, serve API |
| `RuleClassifier` | State classification | occupancy, flow, spacing, wait | state + reasons[] | Persist state |
| `CongestionStopwatch` | Duration tracking | state transitions | duration_s + events | Classify |
| `SnapshotBuilder` | Serialization | DashboardModel + tracks + events | QueueSnapshot (frozen) | Mutate inputs |
| `SnapshotHolder` | Thread-safe sharing | QueueSnapshot + frame | Latest snapshot/frame | Process data |
| `FastAPI` | HTTP + MJPEG serving | SnapshotHolder | JSON / MJPEG stream | Compute analytics |
| `React Dashboard` | Presentation | API responses | UI | Compute queue metrics |

## Key Data Structures

### QueueSnapshot (Frozen Dataclass)
```python
@dataclass(frozen=True)
class QueueSnapshot:
    timestamp: str                    # ISO8601 UTC
    frame_id: int
    freshness: Literal["LIVE","STALE","OFFLINE"]
    staleness_ms: int
    fps: float
    inference_ms: float
    tracking_ms: float
    analytics_ms: float
    detection_count: int
    tracked_people: int
    people_in_queue: int
    total_in: int
    total_out: int
    queue_state: Literal["NORMAL","BUSY","CONGESTED","BACKLOG","DATA_UNAVAILABLE"]
    occupancy_percent: float
    entry_rate: float
    exit_rate: float
    avg_wait_s: float
    longest_wait_s: float
    congestion_duration_s: float
    spacing_px: Optional[float]
    movement: Literal["MOVING","STAGNANT","IDLE"]
    config_status: ConfigStatus
    calibration: Literal["METRES","PX ONLY"]
    camera_state: CameraState
    tracks: List[Track]
    events: List[Event]
```

### Track (Normalized 0..1)
```python
@dataclass(frozen=True)
class Track:
    id: int
    x: float      # center x / width
    y: float      # center y / height
    w: float      # box width / width
    h: float      # box height / height
    in_roi: bool
    confidence: float = 1.0
    age: int = 0
    hits: int = 0
    state: str = "active"
```

### Event
```python
@dataclass(frozen=True)
class Event:
    id: int
    t: int        # epoch ms
    kind: Literal["entry","exit","state","congestion","system"]
    severity: Literal["info","warn","critical","good"]
    label: str
    detail: Optional[str]
```

## Threading Model

```
Main Thread (detect_people.py)
├── FrameStream (background thread) → produces frames
├── YOLO inference
├── SORT tracking
├── QueueAnalytics.update()
├── SnapshotBuilder.build()
├── SnapshotHolder.update() ◄─── writer (holds lock briefly)
├── draw_dashboard() / draw_overlay()
└── cv2.imshow() + cv2.waitKey(1)

FastAPI Thread Pool (uvicorn workers)
├── GET /api/snapshot → SnapshotHolder.get_latest_snapshot() ◄─── reader
├── GET /api/health → SnapshotHolder.get_latest_snapshot()
├── GET /api/events → SnapshotHolder.get_latest_snapshot()
├── GET /api/config → loads camera.yaml (no holder)
└── GET /api/camera/stream → SnapshotHolder.get_latest_frame() ◄─── reader
```

## Configuration Flow

```
config/camera.yaml (single source of truth)
         │
         ▼
load_camera_config() → CameraConfig dataclass
         │
         ├──► QueueAnalytics (thresholds, ROI, line, capacity)
         ├──► SnapshotBuilder (normalization, config_status)
         ├──► draw_dashboard / draw_overlay (layout, colors)
         ├──► FastAPI /api/config → JSON response
         └──► --define-roi → writes back to camera.yaml
```

## Four Queue Concepts (Kept Separate)

| Concept | Definition | Computed By | Units |
|---------|------------|-------------|-------|
| **Occupancy** | People with centroid inside ROI | `QueueAnalytics` | count / % |
| **Throughput / Flow** | People crossing EXIT line per time | `QueueAnalytics` | people/min |
| **Waiting** | Entry → exit duration per track | `QueueAnalytics` | seconds |
| **Congestion** | Rule-based state from multiple variables | `RuleClassifier` | state enum |

> **Critical:** These four concepts are computed independently. Congestion is NOT just "high occupancy" — it requires sustained occupancy + spacing confirmation + flow conditions.

## Legacy Multi-Sensor Engine (Dormant)

The `core/` directory contains a complete but dormant multi-sensor engine (Sensor 1/2/3, fusion, alerts, people/m², MQTT-era). It is **not part of the current runtime** and only activates with `--monitor` flag.

| Module | Purpose | Status |
|--------|---------|--------|
| `core/monitor.py` | QueueMonitor (sensor ingestion, emission) | Dormant |
| `core/fusion.py` | Sensor fusion logic | Dormant |
| `core/backlog.py` | Backlog detection | Dormant |
| `core/alerts.py` | Alert engine | Dormant |
| `core/state_machine.py` | Zone state machine | Dormant |
| `core/simulation.py` | Sensor simulation | Dormant |

> Do not modify `core/` unless explicitly working on the future sensor-fusion phase.