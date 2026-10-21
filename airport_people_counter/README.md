# Airport People Counter — Backend Quick Reference

> **Full documentation:** [../docs/](../docs/)

## Quick Start

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
| `--monitor` | off | **LEGACY** Enable core/ engine |

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
make test          # All 262 tests (unittest discover)
make smoke         # Offline image test
./run.sh --test-camera  # Camera FPS test
```

Both runners collect the same suite: `make test` (unittest discover, 262
tests) and `venv/bin/python -m pytest tests/` (294 tests — it also
collects the pytest-style cases unittest skips).

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
- **Backend = Source of Truth** — Dashboard only presents API data
- **Simulator ≠ Production** — "Simulator" source is synthetic data only

---

**Last reviewed against code:** 2026-09-02