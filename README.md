# Airport X-Ray Queue Monitoring System

> Real-time queue analytics for an airport X-ray checkpoint, built on a standard
> camera feed: detect people (YOLOv5s), track them (SORT), and turn their
> movement into occupancy, entry/exit flow, waiting time, and congestion state —
> all local, no cloud, no facial recognition.

**The problem it solves:** staff can't judge queue conditions reliably by eye —
no early warning, no trend, no record. This system answers *“how many are
waiting, is it getting worse, how long is the wait, is the camera healthy?”* at
a glance. Full product definition: [docs/PRD.md](docs/PRD.md).

## Features

- Person detection (YOLOv5s) + multi-object tracking (SORT) with stable IDs
- Configurable queue ROI and **separate IN/OUT counting lines** with
  jitter-proof crossing debounce and direction handling
- Metrics: occupancy %, entry/exit flow (1 & 5 min), per-person waiting time,
  pixel spacing, movement, congestion state machine (EMPTY → … → CONGESTED)
- **Interactive calibration from the dashboard** — draw geometry on the live
  feed (“Edit mapping”), saved per camera in `config/mappings.yaml` with
  atomic writes and instant hot reload
- Live operator dashboard (React): status, camera, key-metric tiles, charts,
  waiting-time panel, activity log, diagnostics drawer, LIVE + DEMO modes
- Event/samples recording (JSONL), offline accuracy evaluator, and a full
  dataset/training tooling chain for a future custom model

## Tech Stack

| Layer | Stack |
|-------|-------|
| Vision | Python 3.9, YOLOv5s (vendored), SORT, OpenCV, PyTorch (MPS/CPU) |
| API | FastAPI + Uvicorn, Pydantic, MJPEG streaming |
| Dashboard | React 19, TypeScript, Vite 8, Tailwind CSS v4, Vitest |
| Config | `airport_people_counter/config/camera.yaml` (thresholds), `config/mappings.yaml` (geometry) |
| Data | in-memory history buffer (60 min @ 1 Hz), JSONL exports, SQLite catalog for ML datasets |

## Quick Start

**Prerequisites:** Python 3.9, Node 18+, pnpm (`corepack enable`), camera
permission for your terminal (macOS: Settings → Privacy → Camera).

```bash
# One command — pipeline (headless + API :8080) + dashboard (:8443)
./start.sh
# then open http://localhost:8443  (Ctrl+C stops everything)
```

Manual alternative:

```bash
# Terminal 1
cd airport_people_counter && ./run.sh --headless --api --source 0
# Terminal 2
cd Queue-monitoring-dashboard && pnpm dev
```

Useful: `./start.sh --source clip.mp4` (video),
`./run.sh --list-cameras`, `./run.sh --test-camera 0`,
dashboard **DEMO** mode needs no camera.

### Environment & ports

| Variable / flag | Default | Purpose |
|-----------------|---------|---------|
| `PORT` | 8443 | Dashboard dev-server port |
| `API_PROXY_TARGET` | `http://localhost:8080` | Where Vite proxies `/api` |
| `--api-port` / `--api-host` | 8080 / 0.0.0.0 | Backend bind |

No `.env` files, no secrets — everything runs on localhost.

## Configuration

- **Behaviour** (thresholds, capacity, debounce, flow windows, backlog) —
  `airport_people_counter/config/camera.yaml`.
  Full schema: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).
- **Geometry** (queue ROI + IN/OUT lines, normalised per camera) —
  press **“Edit mapping”** on the dashboard’s Live camera panel (preferred), or
  hand-edit `airport_people_counter/config/mappings.yaml`. A valid mapping is
  the single geometry source; legacy pixel geometry in `camera.yaml` is only a
  fallback (`--define-roi` still works).
- Classification thresholds are **example values** — calibrate against your own
  footage before trusting states.

## Verify an Installation

```bash
cd airport_people_counter && make test        # unit tests (all green)
cd .. && cd Queue-monitoring-dashboard && pnpm test   # frontend tests

curl -s localhost:8080/api/dashboard/health            # OK when running
curl -N --max-time 3 localhost:8080/api/camera/stream | wc -c   # bytes flow
```

Accuracy gates and manual scenarios: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Documentation

| File | Purpose |
|------|---------|
| [docs/PRD.md](docs/PRD.md) | Product requirements, feature status (implemented / partial / planned) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the system fits together, key decisions |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, commands, testing, validation, debugging |
| [docs/API.md](docs/API.md) | Endpoint reference and data models |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | `camera.yaml` + mapping schema |
| [docs/DATASETS.md](docs/DATASETS.md) | Dataset, training, and evaluation workflow |
| [AGENTS.md](AGENTS.md) | Instructions for AI coding agents |

## Scope Boundaries

Deliberately **not** built: physical sensors (ultrasonic/ToF/LiDAR), sensor
fusion / MQTT / cloud, facial recognition or passenger identification,
production-grade accuracy claims (accuracy gates not yet passed on real
footage), and a multi-lane UI. Justification and status: [docs/PRD.md §7](docs/PRD.md).

---

Internal project — not published.
