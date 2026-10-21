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
| `QueueAnalytics` | All queue metrics; resolves the active mapping geometry (mappings.yaml → legacy fallback) | tracks[], frame_shape, confs | Snapshot dict + events | Render, serve API |
| `RuleClassifier` | State classification | occupancy, flow, spacing, wait | state + reasons[] | Persist state |
| `queue_prototype._handle_crossing` | Per-track/per-line crossing engine: side commit, debounce, dedup, reversed crossings | ref point, line geometry | line events, counter updates | Double-count, lose jitter |
| `mapping` (app/mapping.py) | Mapping schema, validation, normalised↔pixel conversion, atomic persistence | MappingGeometry / camera.yaml | errors / PixelGeometry / mappings.yaml | Classify, detect |
| `reload_signal` (app/reload_signal.py) | Process-local "config changed" flag from the API thread | request_config_reload() | consume→reload | Persist anything |
| `MappingEditor` (frontend) | Draw/correct ROI + IN/OUT lines over the raw stream | GET/PUT /api/dashboard/mapping | mapping payload | Enforce validation (backend does) |
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
├── consume_config_reload() → hot-reload config + reset crossing state
└── cv2.imshow() + cv2.waitKey(1)

FastAPI Thread Pool (uvicorn workers)
├── GET /api/snapshot → SnapshotHolder.get_latest_snapshot() ◄─── reader
├── GET /api/health → SnapshotHolder.get_latest_snapshot()
├── GET /api/events → SnapshotHolder.get_latest_snapshot()
├── GET /api/config → loads camera.yaml (no holder)
├── GET/PUT /api/dashboard/mapping → load_camera_config + atomic persist
├── GET /api/camera/stream → holder annotated OR raw frame (?annotated=0)
└── PUT success → reload_signal.request_config_reload() ──► main thread picks it up on its next frame
```

> Note: image sources (`*.jpg`) take a single-shot path that stops before
> the frame loop — API streaming and hot reload are video/camera features.

## Configuration Flow

```
config/mappings.yaml (per-camera mapping — geometry source of truth
                      while a valid entry exists)
         │ validate + normalise (app/mapping.py)
         ▼
load_camera_config() → CameraConfig(mapping=…, mapping_source=…)
         │
         ├── valid mapping found  → mapping drives analytics (single source)
         ├── file absent/invalid  → legacy fallback derived in-memory from
         │                          camera.yaml roi/counting_line/xray_side_point
         └── neither              → geometry unconfigured
         │
         ├──► QueueAnalytics (mapping reference point + crossings + ROI)
         ├──► SnapshotBuilder (config_status.mapping_active/reference_point)
         ├──► draw_dashboard / draw_overlay (mapping geometry)
         ├──► FastAPI /api/dashboard/{config,mapping} → JSON responses
         └──► PUT /api/dashboard/mapping → validate → atomic save (.bak)
                └──► reload_signal → detection loop re-reads + reset_mapping()

config/camera.yaml (behaviour thresholds; geometry = legacy fallback)
         │
         ├──► QueueAnalytics (thresholds, capacity, crossing_debounce_frames)
         ├──► SnapshotBuilder (normalization, config_status)
         ├──► draw_dashboard / draw_overlay (layout, colors)
         └──► --define-roi → writes back to camera.yaml
```

## Camera Line Mapping & Calibration

Resolution independence: mapping coordinates are normalised (`x`, `y` ∈
`[0, 1]`) against the frame the geometry was drawn on, and converted to
pixels of whatever the pipeline actually delivers
(`mapping.to_pixel(mapping, (w, h))`) — recalibration survives a camera
resolution change as long as the scene framing matches.

Crossing engine (per track, per line — `PersonState.line_states`):

1. Reference point: `bottom_center` of the bbox by default (see
   `reference_point`), legacy `centroid` fallback.
2. First sighting commits the side (sign of `_line_side`).
3. A side change only *proposes* a crossing; it is committed after
   `crossing_debounce_frames` consecutive frames on the far side
   (default 2 ⇒ crossing frame + 1 stability frame). A shorter jitter
   re-entry cancels it; standing still can never duplicate an event.
4. A committed crossing is **forward** when the track moves onto the
   half-plane the line's direction points into (`IN` = enter the queue,
   `OUT` = exit towards X-ray), else `line_crossing_reversed` — explicit
   and never counted, so walking back out cannot inflate throughput.
5. Counters: forward IN registers entry only; forward OUT finalises
   `exit_unix`/waiting time. Occupancy stays live ROI membership — it is
   never entries-minus-exits, and only OUT crossings evict a track.

Line-ordering contract for legacy geometry: the OUT line's positive
half-plane (sign of `_line_side`) must contain the `xray_side_point`.
Since the sign depends on endpoint order (a horizontal and a vertical
counting line flip it), `from_legacy_camera_config` orients the
derived IN/OUT lines by the sign at the X-ray point — it is never
assumed from the point order.

Four Queue Concepts (Kept Separate)

| Concept | Definition | Computed By | Units |
|---------|------------|-------------|-------|
| **Occupancy** | People with reference point inside ROI (bottom_center/centroid) — live membership, never entries-minus-exits | `QueueAnalytics` | count / % |
| **Throughput / Flow** | People crossing the OUT line (forward) per time | `QueueAnalytics` | people/min |
| **Waiting** | Entry (ROI join or forward IN) → forward OUT exit duration per track | `QueueAnalytics` | seconds |
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