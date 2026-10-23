# Architecture

How the Airport X-Ray Queue Monitoring System fits together. For requirements
and status see [PRD.md](PRD.md); for wire formats see [API.md](API.md); for
config schema see [CONFIGURATION.md](CONFIGURATION.md).

## System Overview

A **camera-only prototype**: frames → person detection (YOLOv5s) → tracking
(SORT) → queue analytics (ROI, lines, classifier) → snapshot → FastAPI → React
dashboard. One process runs the pipeline, a second thread pool serves HTTP;
they share data through a thread-safe holder. Everything is local — there are
**no external service integrations** and **no runtime database** (history lives
in an in-memory ring buffer; only the ML dataset catalog uses SQLite).

```
camera / video / image / RTSP
        │  FrameStream (background thread, paced to --fps)
        ▼
YOLOv5s (COCO person) ──► SORT (stable IDs, Kalman + IoU)
        │ detections                    │ tracks
        ▼                               ▼
        └────────► QueueAnalytics ◄─────┘
                    │  • ROI membership → occupancy (ref point: bottom_center)
                    │  • IN/OUT crossings → entries, exits, waiting, flow
                    │  • RuleClassifier → state + reasons[]
                    │  • CongestionStopwatch, event log
                    ▼
        SnapshotBuilder (frozen dataclass) ──► draw_dashboard() [OpenCV window]
                    │
                    ▼
        SnapshotHolder (thread-safe singleton)
             │ writer: pipeline per frame
             │ readers: FastAPI routes, MJPEG encoder
             ▼
        FastAPI :8080  ── /api/* , /api/dashboard/* , /api/camera/stream
             │ Vite proxy (/api → :8080)
             ▼
        React dashboard :8443  (LIVE mode = API, DEMO mode = simulator)
```

## Module Responsibilities

| Module | Responsibility | Must NOT |
|--------|----------------|----------|
| `app/detect_people.py` | CLI, FrameStream capture, YOLO inference, main loop, hot-reload polling, rendering | Own queue business logic |
| `app/sort_tracker.py` | Tracking (Kalman + IoU + Hungarian) | Detect, classify |
| `app/queue_prototype.py` | QueueAnalytics, RuleClassifier, crossing engine, config loading, HUD | Render, serve HTTP |
| `app/mapping.py` | Mapping schema, validation, normalised↔pixel conversion, atomic persistence | Detect, classify |
| `app/reload_signal.py` | In-process “config changed” flag set by API writes | Persist anything |
| `app/snapshot.py` | Frozen dataclasses + SnapshotBuilder (internal → JSON-ready) | Mutate inputs |
| `app/snapshot_holder.py` | Thread-safe snapshot/frame sharing incl. raw frame slot | Compute |
| `app/snapshot_buffer.py` | 1 Hz history ring buffer (3600 samples) + downsampling | Serve HTTP |
| `app/api_server.py`, `api_models.py` | HTTP routes, Pydantic contracts, MJPEG | Compute analytics |
| `app/shutdown.py` | Cooperative shutdown across loop + server | — |
| `app/ml/*`, `scripts/*` | Dataset contract/manifests/split/catalog, training + evaluation tooling | Runtime analytics |
| `Queue-monitoring-dashboard/src` | Presentation only (poll 1 Hz, render) | Recompute queue metrics |

## Key Decisions (verified)

| Decision | Why |
|----------|-----|
| Backend is the single source of truth for queue metrics | Dashboard must not duplicate classifier/flow/wait logic |
| Mapping geometry is **normalised** (`x, y ∈ [0,1]`) per camera | Survives resolution changes; one store for any frame size (`config/mappings.yaml`) |
| One active geometry source (mapping wins; legacy `camera.yaml` geometry is in-memory fallback; never combined) | Two overlapping geometries would double-apply semantics |
| Separate IN and OUT lines with explicit direction; reversed crossings are events, not counts | Walk-back across a line must never inflate throughput |
| Crossings commit only after N far-side frames (per track, per line) | Camera jitter would otherwise double-count; knob is separate from state-machine debounce |
| Snapshot as frozen dataclasses, shared via one locked holder | Pipeline writes at frame rate; HTTP readers must never see partial state |
| Validation before persistence + atomic `os.replace` + `.bak` | A bad payload can never replace a working mapping |
| Raw stream is opt-in (`?annotated=0`) | Existing annotated-stream consumers keep their burn-ins |

## Data Flow: Backend → Dashboard

1. Pipeline builds a `QueueSnapshot` per frame (timestamp, freshness, queue
   metrics, config status, normalised `tracks[]`, `events[]`).
2. Dashboard polls `GET /api/dashboard/snapshot` at 1 Hz; history comes from
   the ring buffer (1 Hz × 60 min, downsampled ≤ 240 points per response,
   `?points` up to 3600).
3. Camera panel renders `GET /api/camera/stream` (MJPEG) and overlays are
   optional; the mapping editor uses `?annotated=0`.
4. Simulator/DEMO mode bypasses all of this with synthetic telemetry.

Field-level contracts: [API.md](API.md).

## Classifier State Machine

Occupancy ladder (thresholds from `camera.yaml` → [CONFIGURATION.md](CONFIGURATION.md)):

```
EMPTY (<10%) ──► SPARSE (<40%) ──► NORMAL (<60%) ──► CROWDED (<80%)
                                                            │  ≥80% AND spacing ≤ 60px
                                                            ▼
                                                       CONGESTED
                                                            │  sustained 10+ s with
                                                            │  configured flow baseline
                                                            ▼
                                       BACKLOG (disabled by default — needs baseline)
```

- **Hysteresis:** downgrade threshold = threshold × (1 − `hysteresis`, default 0.05).
- **Debounce:** a candidate state must persist `debounce_frames` (5) frames.
- CONGESTED requires mean spacing ≤ `congested_max_spacing_px` (60) when
  `congested_require_spacing` is true.
- Every snapshot carries the exact `reasons[]` that produced the state.

## Threading Model

```
Main thread (detect_people.py)
├── FrameStream thread → frame queue
├── YOLO → SORT → QueueAnalytics.update()
├── SnapshotBuilder → SnapshotHolder.update(snap, frame, raw_frame)
├── draw_dashboard() / cv2.waitKey(1)
└── consume_config_reload() each frame → reload config + reset crossing state

FastAPI thread pool (uvicorn)
├── readers → SnapshotHolder / history buffer
├── GET/PUT /api/dashboard/mapping → load config, atomic persist
└── successful PUT → reload_signal.request_config_reload() ──► main thread
```

Caveat: image sources (`*.jpg`) take a single-shot path that never enters the
frame loop — API streaming and hot reload require a video/camera source.

## Configuration Flow

```
config/mappings.yaml ── validate (app/mapping.py) ──┐
                                                    ├─► exactly one geometry source
config/camera.yaml legacy roi/line/xray_side_point ─┘   (mapping wins; legacy is
        │                                                in-memory fallback)
        ├─ thresholds, capacity, debounce, recommendations,
        │  calibration, movement, backlog  ──────────────► QueueAnalytics
        └─ --define-roi writes legacy geometry back

PUT /api/dashboard/mapping ─► validate ─► atomic save (.bak) ─► reload_signal
failed validation ─► nothing written, running mapping untouched
```

## Camera Line Mapping & Calibration

- Coordinates are normalised against the frame they were drawn on and
  converted per delivered resolution (`mapping.to_pixel`), so calibration
  survives a resolution change when framing is unchanged.
- Crossing engine (per track, per line): first sighting commits the side; a
  side change only *proposes* a crossing, committed after
  `crossing_debounce_frames` consecutive far-side frames; forward direction
  commits counters (`IN` = entry, `OUT` = exit/wait), the opposite direction
  emits `line_crossing_reversed` (never counted). Occupancy is always live
  ROI membership — never entries-minus-exits.
- Legacy conversion orients derived IN/OUT lines by the sign of the side test
  at `xray_side_point`, so orientation never depends on endpoint order.
- UI: dashboard “Edit mapping” → `MappingEditor.tsx` (letterboxed raw feed,
  draggable/keyboard handles, validation display).

## Four Queue Concepts (Kept Separate)

| Concept | Definition | Units |
|---------|------------|-------|
| Occupancy | People whose reference point is inside the ROI — live membership | count / % |
| Throughput / flow | Forward OUT crossings per minute | people/min |
| Waiting | Entry (ROI join or forward IN) → forward OUT exit, per track | seconds |
| Congestion | Rule state from occupancy + spacing + flow + wait (sustained) | enum |

Congestion is *not* “high occupancy” alone.

## Storage & External Integrations

- **Runtime DB:** none. Snapshot history is an in-memory ring buffer
  (3600 × 1 s); events/samples can be written to JSONL via `--record`,
  `--record-samples`.
- **ML catalog:** SQLite under `airport_people_counter/datasets/` (dataset
  versions, manifests, experiments) — see [DATASETS.md](DATASETS.md).
- **External services:** none (no cloud, no MQTT, no auth providers). Network
  surface = FastAPI on localhost + Vite dev proxy.
