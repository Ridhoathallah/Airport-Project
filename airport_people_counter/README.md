# Airport People Counter — Backend Quick Reference

> **Full documentation:** [../docs/](../docs/)

## Quick Start

The whole system (camera pipeline + API + dashboard) starts with one
command from the project root:

```bash
cd ..
./start.sh                 # camera 0, dashboard on http://localhost:8443
./start.sh --source 1      # a different camera
./start.sh --source clip.mp4
```

To run only the backend:

```bash
# Webcam + API (port 8080)
./run.sh --api

# Webcam only
./run.sh

# Video file
./run.sh --source input/video.mp4

# Interactive ROI setup
./run.sh --define-roi

# Preview saved layout
./run.sh --roi-preview

# Run tests
make test
```

## Key Files

| File | Purpose |
|------|---------|
| `app/detect_people.py` | Entry point: vision + CLI + rendering loop |
| `app/queue_prototype.py` | Analytics: config, metrics, classifier, HUD |
| `app/sort_tracker.py` | SORT tracker (Kalman + IoU) |
| `app/snapshot.py` | Immutable snapshot dataclasses + builder |
| `app/snapshot_holder.py` | Thread-safe snapshot + frame holder |
| `app/api_models.py` | Pydantic models for FastAPI |
| `app/api_server.py` | FastAPI endpoints + MJPEG stream |
| `config/camera.yaml` | Runtime configuration (single source of truth) |
| `scripts/evaluate_counts.py` | Offline evaluation (occupancy vs ground truth) |

## Common Commands

| Command | Description |
|---------|-------------|
| `./run.sh --api` | Webcam + FastAPI server (port 8080) |
| `./run.sh --define-roi` | Interactive ROI + counting line setup |
| `./run.sh --roi-preview` | Preview saved layout |
| `./run.sh --source video.mp4` | Run on video file |
| `./run.sh --record out.jsonl` | Record events (JSONL) |
| `./run.sh --record-samples out.jsonl` | Record 1 Hz occupancy samples |
| `./run.sh --list-cameras` | List available cameras |
| `./run.sh --test-camera` | Test camera FPS (no YOLO) |
| `make test` | Run 262 unit tests |
| `make smoke` | Quick test on sample image |

## CLI Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--source` | config or `0` | Webcam index, video path, or RTSP/HTTP URL |
| `--api` | off | Enable FastAPI server (port 8080) |
| `--api-port` | `8080` | API server port |
| `--conf` | `0.50` | Detection confidence threshold |
| `--imgsz` | `480` | YOLO inference size (px) |
| `--device` | `mps`/`cpu` | `mps` \| `cpu` \| `cuda` |
| `--define-roi` | off | Interactive ROI + line setup |
| `--roi-preview` | off | Preview saved layout |
| `--debug` | off | Show classifier reasons on HUD |
| `--legacy-hud` | off | Use legacy full-frame HUD |

## Configuration

All runtime config in `config/camera.yaml`:

```yaml
camera:
  camera_id: "CAMERA 01"
  source: 0
  roi: null              # Set via --define-roi
  counting_line: null    # Set via --define-roi
  xray_side_point: null  # Set via --define-roi
  queue_capacity: 30
classifier:
  occupancy_sparse: 0.10
  occupancy_normal: 0.40
  occupancy_crowded: 0.60
  occupancy_congested: 0.80
  hysteresis: 0.05
  debounce_frames: 5
```

## API Endpoints (when `--api` enabled)

| Endpoint | Description |
|----------|-------------|
| `GET /api/snapshot` | Latest queue snapshot (1 Hz) |
| `GET /api/health` | Camera/CV pipeline health |
| `GET /api/config` | Camera configuration (read-only) |
| `GET /api/events` | Recent events |
| `GET /api/camera/stream` | MJPEG stream |
| `GET /api/dashboard/snapshot` | Dashboard snapshot: queue, system, camera, history, events |
| `GET /api/dashboard/config` | Dashboard configuration (ROI, counting line, calibration) |
| `GET /api/dashboard/health` | Dashboard health status |
| `GET /api/dashboard/history?points=N` | Occupancy history, N evenly spaced samples (2–3600) |
| `POST /api/shutdown` | Stop the pipeline and the API server |

### Dashboard payload notes

| Field | Meaning |
|-------|---------|
| `queue.action` | Operational guidance for the current condition. Taken from the config-overridable `recommendations:` table, never invented by the dashboard. |
| `queue.reason` | Why the classifier produced this state. |
| `events[].label` / `events[].detail` | Operator-facing wording. Internal state names are translated before they leave the backend. |
| `events[].technical` | The engineering description of the same event, for the diagnostics view only. |
| `config.wait_alert_seconds` | The wait duration the detector treats as long enough to contribute to a backlog condition. Used as the waiting-time threshold marker. |

### History sampling

Occupancy history is sampled on a fixed **1 Hz** grid and kept for 60 minutes
(3600 samples). The polled snapshot carries an evenly spaced view of it (at
most 240 points, one per 15 s), so the dashboard's 5m / 15m / 30m / 60m
ranges describe real elapsed time rather than a number of frames.

## Architecture

```
Camera → YOLOv5s → SORT → QueueAnalytics → SnapshotBuilder → SnapshotHolder
                                                                    ↓
                                                              FastAPI (8080)
                                                                    ↓
                                                              Vite Proxy (8443)
                                                                    ↓
                                                              React Dashboard
```

## Testing

```bash
make test          # Every test: unittest discover + pytest
make smoke         # Offline image test
./run.sh --test-camera  # Camera FPS test
```

The suite is written in two styles, so `make test` invokes both runners:
`unittest discover` (308 `TestCase` tests) and `pytest` (358 tests, which
also collects the pytest-style function tests unittest skips: the history
buffer, the dashboard adapter, the dashboard API, the dashboard models and
the event language). `make test-unittest` and `make test-pytest` run one
runner each. pytest is optional: if it is not installed, `make test` runs
the unittest suite and says so.

## ML Dataset & Experiment Tooling

`ml/` and `scripts/` hold the dataset lifecycle (contract, validation,
leakage-safe splitting, manifests, metadata catalog, experiment
provenance, evaluation). It is independent of the runtime application and
of the runtime monitoring database.

```bash
make catalog-init        # create the metadata catalog (idempotent)
make dataset-validate    # validate datasets/people_v1, write a JSON report
make dataset-build       # validate -> split -> manifest -> catalog
make dataset-check       # training preflight for a dataset (runs no training)
make catalog-stats       # row counts per catalog table
```

See [../docs/DATASETS.md](../docs/DATASETS.md) for the full workflow.

## Important Boundaries

- **No facial recognition** — Tracks people, doesn't identify them
- **Camera-only prototype** — No sensor fusion, no cloud, local-only
- **Backend = Source of Truth** — Dashboard only presents API data; a
  camera failure keeps the API alive (bounded reconnect) and reports
  honest unavailable states instead of fabricated numbers

---

**Last reviewed against code:** 2026-09-02