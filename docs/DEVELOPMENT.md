# Development Guide

Practical guide: setup, commands, testing, validation, debugging. Architecture
lives in [ARCHITECTURE.md](ARCHITECTURE.md); API contracts in
[API.md](API.md); config schema in [CONFIGURATION.md](CONFIGURATION.md);
dataset workflow in [DATASETS.md](DATASETS.md); product status in
[PRD.md](PRD.md).

## Prerequisites

- **Python 3.9** (developed on 3.9.6); Apple Silicon uses MPS automatically
- **Node.js 18+** (repo developed on 22.x) and **pnpm** (via corepack:
  `corepack enable && corepack prepare pnpm@10.34.3 --activate`)
- A camera (webcam / RTSP) or a video file; a real camera is optional for the
  test suite (synthetic tracks only)
- macOS camera permission must be granted to your terminal (System Settings →
  Privacy & Security → Camera, then relaunch the terminal)

## Installation 

```bash
# Backend — run.sh auto-creates the venv and installs dependencies
cd airport_people_counter && ./run.sh --help

# Or manually
python3.9 -m venv venv
venv/bin/pip install -r requirements/requirements.txt

# Dashboard
cd Queue-monitoring-dashboard && pnpm install
```

## Running the Application

```bash
# From the project root — starts pipeline (headless + API) and dashboard
./start.sh                      # camera 0, dashboard http://localhost:8443
./start.sh --source clip.mp4    # video file
./start.sh --no-dashboard       # backend only
./start.sh --lan                # dashboard exposed on the LAN
Ctrl+C                          # stops both (reuses already-running services)
```

The launcher prints a status block (`YOLO / Camera / FastAPI / Dashboard`)
where `Camera` honestly reports `NO SIGNAL` when no frames arrive.

By default the dashboard is reachable on **localhost only**; pass `--lan`
to expose it on the network.

Service output is written to `.run/backend.log` and `.run/frontend.log`.
The pipeline prints one status line per second to that log (the live
per-frame line is only used on a terminal). Logs are rotated to a single
`.1` file once they pass 8 MB, so a long-running deployment cannot fill
the disk.

Manual (two terminals):

```bash
# Terminal 1 — backend
cd airport_people_counter
./run.sh --headless --api --source 0        # or: make api

# Terminal 2 — dashboard
cd Queue-monitoring-dashboard
pnpm dev                                     # http://localhost:8443
```

The dashboard polls the backend at 1 Hz (proxied `/api` → `localhost:8080`);
with no backend reachable it shows an explicit "Backend unreachable" state
rather than inventing data. Image sources
(`*.jpg`) are single-shot: streaming/hot-reload need camera or video sources.

### Useful backend commands

| Command | Purpose |
|---------|---------|
| `./run.sh --list-cameras` | Indices, resolution, measured FPS, device names |
| `./run.sh --test-camera 0` | Camera FPS test without YOLO |
| `./run.sh --define-roi` | Interactive legacy ROI/line setup (writes `camera.yaml`) |
| `./run.sh --roi-preview` | Preview saved legacy layout |
| `./run.sh --record output/events.jsonl` | Record events (JSONL) |
| `./run.sh --record-samples output/samples.jsonl` | Record 1 Hz occupancy samples |
| `./run.sh --conf 0.3 --imgsz 320 --fps 15` | Confidence / speed knobs |
| `./run.sh --debug` / `--legacy-hud` | HUD variants |
| `make run api test smoke …` | Makefile shortcuts (`make help`) |

Full CLI: `./run.sh --help` (the `detect_people.py --help` flag list is the
authority — do not duplicate it here).

### Ports & environment variables

| Item | Default | Override |
|------|---------|----------|
| FastAPI | `0.0.0.0:8080` | `--api-host`, `--api-port` |
| Dashboard (Vite) | `8443` | `PORT` env var |
| Vite `/api` proxy | `http://localhost:8080` | `API_PROXY_TARGET` env var |
| Vite base URL | `/` | `FIGMA_PUBLIC_URL` (Figma hosting only) |

No `.env` files are used and no secrets are configured anywhere.

## Repository Structure

```
Airport-Project/
├── start.sh                     # Unified launcher (backend + dashboard)
├── README.md, AGENTS.md         # Entry point + agent instructions
├── docs/                        # This documentation set
├── airport_people_counter/
│   ├── app/                     # Runtime: detect_people, queue_prototype,
│   │                            #   mapping, sort_tracker, snapshot*, api_*, ml/
│   ├── config/camera.yaml       # Behaviour thresholds + legacy geometry
│   ├── config/mappings.yaml     # Per-camera mapping geometry (editor-written)
│   ├── tests/                   # unittest + pytest suites
│   ├── ml/ scripts/             # Dataset, training, evaluation tooling
│   ├── datasets/                # Dataset versions, manifests, SQLite catalog
│   ├── models/                  # yolov5s.pt, YuNet onnx
│   ├── yolov5/                  # Vendored YOLOv5 repo
│   ├── requirements/, Makefile, run.sh
└── Queue-monitoring-dashboard/  # React 19 + Vite 8 + Tailwind v4
    ├── src/App.tsx              # Layout, theme toggle
    ├── src/components/          # sections, charts, health, MappingEditor
    ├── src/lib/                 # api.ts, telemetry.ts, mapping.ts (+tests)
    ├── vite.config.ts, package.json, vitest.config.ts
    └── AGENTS.md                # Figma-scaffold-specific agent notes
```

### Authoritative files by subsystem

| Subsystem | File(s) |
|-----------|---------|
| Capture, inference, main loop, CLI | `app/detect_people.py` |
| Tracking | `app/sort_tracker.py` |
| Analytics, classifier, crossings, legacy config | `app/queue_prototype.py` |
| Mapping schema/validation/persistence | `app/mapping.py` |
| Snapshot serialization | `app/snapshot.py` |
| Shared state HTTP side | `app/api_server.py`, `app/api_models.py`, `app/dashboard_models.py` |
| Frontend API contract | `src/lib/api.ts` |
| Frontend data hook (API polling, honest unavailable states) | `src/lib/telemetry.ts` |
| Geometry transforms (editor) | `src/lib/mapping.ts` |
| Dashboard panels | `src/components/sections.tsx` |

## Testing

```bash
cd airport_people_counter

make test                 # unittest discover — 308 tests (2026-10-23)
venv/bin/python -m pytest tests -q      # pytest — 358 tests (2026-10-23)

# Targeted
venv/bin/python -m unittest tests.test_queue_prototype -v
venv/bin/python -m unittest tests.test_mapping -v
venv/bin/python -m pytest tests/test_mapping_api.py -q

# Smoke (offline single image; prints PERSONS: N, mapping INFO)
make smoke
```

```bash
cd Queue-monitoring-dashboard
pnpm test                 # vitest — 40 tests (2026-10-23)
npx tsc --noEmit          # type check
pnpm run build            # production build (must stay green)
```

Test counts drift as tests are added — treat the commands as the contract and
the numbers as “as of” markers.

## Validation & Evaluation

### Automated layers
1. **Unit tests** — synthetic tracks, no camera/YOLO needed (fast, deterministic).
2. **Smoke** — real model loads and processes a fixture image.
3. **Type/build** — `tsc --noEmit` + `pnpm run build`.

### Manual smoke (with camera or video)
| Check | Expected |
|-------|----------|
| `● LIVE` indicator increments | frames arriving |
| YOLO boxes + stable SORT IDs | detection + tracking |
| ROI / lines in dashboard | geometry visible; “Edit mapping” opens |
| ENTER/EXIT events on crossing | counting works |
| Occupancy = in-queue / capacity | `IN-QUEUE` matches visible count |
| State ladder | EMPTY → … → CONGESTED per config |
| `--record` JSONL | valid JSON per line |

### Controlled scenario table (default `camera.yaml`, cap = 30)

Occupancy thresholds are ratios of `queue_capacity` — expected counts derived
from them: SPARSE ≥ 10% (≥3 people), NORMAL ≥ 40% (≥12), CROWDED ≥ 60% (≥18),
CONGESTED ≥ 80% (≥24 **and** mean spacing ≤ 60 px). BACKLOG requires a
configured flow baseline (disabled by default).

### API → dashboard sync
```bash
curl -s localhost:8080/api/dashboard/health
curl -s localhost:8080/api/dashboard/snapshot
curl -N --max-time 3 localhost:8080/api/camera/stream | wc -c            # bytes flow
curl -N --max-time 3 "localhost:8080/api/camera/stream?annotated=0" | wc -c
```
Dashboard must show matching numbers (queue, occupancy, ENTR/EXIT) and zero
console errors (favicon 404 is acceptable).

### Offline evaluation against ground truth
```bash
./run.sh --source input/airport_cctv.mp4 --record-samples output/samples.jsonl
# ground truth CSV: start_s,end_s,count[,note]
venv/bin/python scripts/evaluate_counts.py \
    --samples output/samples.jsonl --ground-truth output/ground_truth.csv \
    --tolerance-s 1.0 --gap-s 10.0 --min-coverage 0.8 --json output/eval_report.json
```
Report: MAE, ±1/±2 accuracy, bias, coverage; exit codes 0 success / 1
insufficient data / 2 malformed input.

**Accuracy gates (defined; never yet passed on real footage):** ≥30 min
representative footage, ≥50 intervals, MAE < 1.0, ±1 > 85%, ±2 > 95%,
|bias| < 0.5, coverage ≥ 80%. Never claim “deployment-grade accuracy”
without these.

### Regression checklist (before commit)
- [ ] `make test`, `pytest`, `pnpm test` all green
- [ ] `tsc --noEmit` + `pnpm run build` green
- [ ] Counting-affecting change → manual camera/video pass above
- [ ] API change → curl checks + dashboard sync
- [ ] Config/API/docs updated in the same commit
- [ ] `git diff` reviewed — no unrelated files included

## Debugging

```bash
./run.sh --api --debug        # classifier reasons on the HUD
./run.sh --test-camera 0      # FPS without YOLO
./run.sh --roi-preview        # saved legacy geometry on live frame
curl -s localhost:8080/api/dashboard/health | python3 -m json.tool
```
Per-frame console line: `DET / IN-QUEUE / ENTR / EXIT / FPS / inf / trk / disp /
STATE / MOV / REASONS`; end-of-run prints measured per-component timings.

| Symptom | Check |
|---------|-------|
| No camera frames | macOS permission, `--list-cameras`, camera busy; launcher shows `NO SIGNAL` |
| Low FPS | `--imgsz`, `--fps`, `camera.display_scale`, camera’s real output rate |
| No detections | model path, `--conf`, `--device`, person class only |
| Tracks unstable | `max_age`, `min_hits`, `iou_threshold` in SortTracker |
| ROI/line not counting | mapping active? (`GET /api/dashboard/mapping`), else legacy `camera.yaml` geometry; mapping wins and is never combined with legacy |
| Spurious/reversed counts | `crossing_debounce_frames`, line direction in the editor (Enter swaps) |
| State stuck | `hysteresis`, `debounce_frames`, `congested_require_spacing` |
| `BACKLOG` never fires | `baseline_people_per_minute` is `null` by design |
| API 502 from dashboard | backend not running / wrong `API_PROXY_TARGET` |
| MJPEG empty | holder has no frame (video/camera source required; image mode is single-shot) |
| Dashboard OFFLINE | run with `--api`, source selector = LIVE |

## Common Tasks

**New analytics metric:** field in `queue_prototype` → compute in
`QueueAnalytics.update()` → `build_dashboard_model()` → `snapshot.py` →
Pydantic (`api_models`/`dashboard_models`) → frontend `api.ts` type → panel →
tests (`make test`).

**New API endpoint:** model in the right `*_models.py` → route in
`api_server.py` → docs/API.md → curl test (+ pytest if logic-bearing).

**New dashboard panel:** component in `sections.tsx` → compose in `App.tsx` →
reads the `Snapshot` prop only (no business logic) → `lib/ui.tsx` styles.

**Frontend API call:** interface + fetch function in `src/lib/api.ts` → hook in
`telemetry.ts` → component.

## Code Style & Conventions

**Python:** type hints on public functions; `Literal` for enum strings;
frozen dataclasses for snapshots; no global mutable state except the
`SnapshotHolder` singleton; validate before persisting.

**TypeScript/React:** interfaces for every API response; hooks for fetching;
components are presentation-only; Tailwind utilities (no custom CSS unless in
`index.css`); double quotes in JSX strings containing apostrophes.

**Git:** docs updated in the same commit as code; run the test suites before
committing; keep user work uncommitted untouched (don’t sweep unrelated files
into a feature commit).
