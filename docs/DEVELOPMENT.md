# Development Guide

## Repository Structure

```
Airport-Project/
├── docs/                           # Documentation (this directory)
├── airport_people_counter/         # Python backend
│   ├── app/                        # Camera-only prototype (current runtime)
│   │   ├── detect_people.py        # Entry point: vision + CLI + rendering
│   │   ├── queue_prototype.py      # Analytics: config, metrics, classifier, HUD
│   │   ├── sort_tracker.py         # SORT tracker (Kalman + IoU)
│   │   ├── snapshot.py             # Immutable snapshot dataclasses + builder
│   │   ├── snapshot_holder.py      # Thread-safe snapshot + frame holder
│   │   ├── api_models.py           # Pydantic models for FastAPI
│   │   └── api_server.py           # FastAPI endpoints + MJPEG
│   ├── core/                       # LEGACY multi-sensor engine (dormant)
│   ├── config/                     # YAML configs
│   ├── models/                     # Model weights (yolov5s.pt, YuNet)
│   ├── scripts/                    # Offline evaluation
│   ├── tests/                      # Unit tests (137 tests)
│   ├── yolov5/                     # Vendored YOLOv5 repo
│   ├── requirements/requirements.txt
│   ├── run.sh                      # Launcher (auto venv)
│   └── Makefile                    # Convenience targets
│
└── Queue-monitoring-dashboard/     # React/Vite frontend
    ├── src/
    │   ├── App.tsx                 # Root: source selector, layout
    │   ├── main.tsx                # React entrypoint
    │   ├── index.css               # Tailwind v4 import
    │   ├── lib/
    │   │   ├── telemetry.ts        # useTelemetry (sim) + useRealTelemetry (API)
    │   │   ├── api.ts              # Typed fetch helpers
    │   │   └── ui.ts               # Formatters, colors, components
    │   └── components/
    │       ├── sections.tsx        # QueueStatusPanel, CameraPanel, KpiStack...
    │       ├── charts.tsx          # Sparkline, OccupancyChart, FlowChart
    │       └── health.tsx          # OperatorHealth, DiagnosticsDrawer
    ├── package.json
    ├── vite.config.ts              # /api proxy → localhost:8080
    └── tsconfig.json
```

## Authoritative Files by Subsystem

| Subsystem | Authoritative Files |
|-----------|---------------------|
| Camera capture | `app/detect_people.py` (FrameStream class) |
| YOLO detection | `app/detect_people.py` (load_model, infer) |
| SORT tracking | `app/sort_tracker.py` (SortTracker) |
| Queue analytics | `app/queue_prototype.py` (QueueAnalytics, RuleClassifier) |
| ROI/Line config | `config/camera.yaml` + `app/queue_prototype.py` (CameraConfig) |
| Classification | `app/queue_prototype.py` (RuleClassifier) |
| HUD/Dashboard rendering | `app/queue_prototype.py` (draw_dashboard, draw_overlay) |
| Snapshot/API serialization | `app/snapshot.py` (SnapshotBuilder) |
| Thread-safe sharing | `app/snapshot_holder.py` (SnapshotHolder) |
| FastAPI endpoints | `app/api_server.py` |
| API contracts | `app/api_models.py` |
| Dashboard telemetry | `src/lib/telemetry.ts` (useRealTelemetry) |
| Dashboard API client | `src/lib/api.ts` |
| CameraPanel (MJPEG + overlays) | `src/components/sections.tsx` |
| Vite proxy config | `vite.config.ts` |

---

## Testing

### Run All Tests
```bash
cd airport_people_counter

# Via launcher (auto venv)
make test

# Direct (venv must be active)
venv/bin/python -m unittest discover -s tests

# With pytest
venv/bin/python -m pytest tests -q
```

### Targeted Test Suites
```bash
# Camera-only analytics (synthetic tracks, no camera/YOLO)
venv/bin/python -m unittest tests.test_queue_prototype -v

# Offline evaluator tests
venv/bin/python -m unittest discover -s tests -p "test_evaluate_counts.py"

# Legacy core/ engine tests (only relevant for --monitor)
venv/bin/python -m unittest tests.test_backlog_alerts -v
venv/bin/python -m unittest tests.test_fusion -v
venv/bin/python -m unittest tests.test_metrics -v
venv/bin/python -m unittest tests.test_monitor -v
venv/bin/python -m unittest tests.test_state_machine -v
```

### Smoke Test
```bash
# Quick offline test on sample image (no camera needed)
make smoke
# Runs: venv/bin/python app/detect_people.py --source output/smoke_test/bus.jpg
```

**Expected:** Processes image once, prints `PERSONS: N STATE: UNCONFIGURED (image mode — no FPS)`, exits cleanly.

---

## Debugging Workflow

### Backend Debugging

#### 1. Enable Debug HUD
```bash
./run.sh --api --debug
```
Shows classifier `reasons[]` directly on OpenCV window.

#### 2. Print Console Metrics
The console line prints every frame:
```
DET: 3 IN-QUEUE: 2 ENTR: 5 EXIT: 3 FPS: 22.1 inf: 34.2ms trk: 0.1ms disp: 3.1ms STATE: NORMAL MOV: MOVING REASONS: queue_occupancy_above_normal_threshold
```

#### 3. Test Camera Without YOLO
```bash
./run.sh --test-camera 0
# Measures FPS, verifies frames changing, no YOLO loaded
```

#### 4. List Cameras
```bash
./run.sh --list-cameras
# Shows index, resolution, reported/measured FPS, macOS device name
```

#### 5. Preview ROI Layout
```bash
./run.sh --roi-preview
# Shows saved ROI + counting line on live frame
```

#### 6. Inspect API Endpoints
```bash
# Health
curl http://localhost:8080/api/health

# Snapshot
curl http://localhost:8080/api/snapshot | python -m json.tool

# Config
curl http://localhost:8080/api/config

# Events
curl http://localhost:8080/api/events
```

#### 7. Test MJPEG Stream
```bash
# Quick test (5 seconds)
curl -N --max-time 5 http://localhost:8080/api/camera/stream | head -5
# Should output MJPEG boundary + JPEG data
```

### Frontend Debugging

#### 1. Dev Server
```bash
cd Queue-monitoring-dashboard
npm run dev
# http://localhost:8443 — hot reload enabled
```

#### 2. Browser DevTools
- **Network tab:** Verify `/api/snapshot` polls (1 Hz), `/api/camera/stream` (MJPEG)
- **Console:** React errors, API fetch errors
- **React DevTools:** Component tree, props, state

#### 3. Source Selector
Top bar dropdown: **"Simulator"** (synthetic data) vs **"Live API"** (real backend)

#### 4. Diagnostics Drawer
Click operator health chip (top right) → shows scenario selector, raw snapshot JSON

#### 5. Build Verification
```bash
npm run build
# Production build to dist/ — verifies TypeScript compiles
```

---

## Adding New Features

### Backend: New Analytics Metric
1. Add field to `QueueMetrics` in `queue_prototype.py`
2. Compute in `QueueAnalytics.update()`
3. Add to `build_dashboard_model()` in `queue_prototype.py`
4. Add to `QueueSnapshot` in `snapshot.py`
5. Add to Pydantic model in `api_models.py`
6. Update `SnapshotBuilder._tracks_to_api()` / `_events_to_api()` if needed
7. Run tests: `make test`

### Backend: New API Endpoint
1. Add Pydantic model to `api_models.py`
2. Add endpoint to `api_server.py`
3. Update CORS if new origin needed
4. Test: `curl http://localhost:8080/api/new-endpoint`

### Frontend: New Dashboard Panel
1. Create component in `src/components/sections.tsx`
2. Import and compose in `App.tsx`
3. Ensure it only reads from `Snapshot` prop (no business logic)
4. Use `lib/ui.ts` formatters/colors for consistency

### Frontend: New API Call
1. Add TypeScript interface to `lib/api.ts`
2. Add fetch function to `lib/api.ts`
3. Use in component via `useEffect` or custom hook

---

## Regression Testing Checklist

Before committing changes affecting queue counting:

### Automated
- [ ] `make test` — all 137 tests pass
- [ ] `make smoke` — offline image test passes

### Manual (with real camera or video)
- [ ] Camera opens, delivers frames (`● LIVE` indicator increments)
- [ ] YOLO detects people (boxes appear on OpenCV window / dashboard)
- [ ] SORT assigns stable IDs (same person keeps same ID across frames)
- [ ] ROI works: people outside ROI are greyed out, not counted
- [ ] Counting line works: ENTER/EXIT events fire correctly
- [ ] Occupancy % = people_in_queue / queue_capacity
- [ ] Flow rates: ENTER/EXIT per minute update
- [ ] State classification: EMPTY → SPARSE → NORMAL → CROWDED → CONGESTED
- [ ] Congestion stopwatch: starts on CONGESTED, stops on exit
- [ ] Events logged: `--record output/events.jsonl` produces valid JSONL
- [ ] API `/api/snapshot` returns valid JSON matching Pydantic schema
- [ ] API `/api/camera/stream` serves MJPEG (browser `<img>` loads)
- [ ] Dashboard "Live API" mode shows MJPEG + overlays
- [ ] Dashboard "Simulator" mode shows synthetic data
- [ ] Source selector switches cleanly between Simulator/API

---

## Common Development Tasks

### Update Dependencies

**Backend:**
```bash
cd airport_people_counter
venv/bin/pip install --upgrade -r requirements/requirements.txt
# Update requirements.txt with new versions
venv/bin/pip freeze | grep -E "torch|numpy|opencv|fastapi|uvicorn|pydantic" > requirements/requirements.txt
```

**Frontend:**
```bash
cd Queue-monitoring-dashboard
npm update
npm audit fix
```

### Add Python Type Hints
- Use `from __future__ import annotations`
- Prefer `Literal` for enums, `Optional` for nullable
- Run `venv/bin/python -m mypy app/` if mypy configured

### Profile Performance
```bash
# Per-component timings printed every 10 frames in console
# End-of-run summary:
# FPS       : 23.6
# Inference : 34.0 ms/frame
# Tracking  : 0.1  ms/frame
# Analytics : 0.1  ms/frame
# Display   : 3.2  ms/frame
# Capture   : 30.0 FPS from source
```

### Modify Classifier Thresholds
Edit `config/camera.yaml`:
```yaml
classifier:
  occupancy_sparse: 0.10
  occupancy_normal: 0.40
  occupancy_crowded: 0.60
  occupancy_congested: 0.80
  hysteresis: 0.05
  debounce_frames: 5
  congested_max_spacing_px: 60
  congested_require_spacing: true
```
> Thresholds are EXAMPLE VALUES — must be calibrated against real CCTV footage.

---

## Code Style & Conventions

### Python
- Type hints on all public functions
- `Literal` for enum-like strings
- Dataclasses for immutable data
- `frozen=True` for snapshot objects
- No global mutable state (except `SnapshotHolder` singleton)

### TypeScript/React
- Interfaces for all API responses
- Custom hooks for data fetching (`useRealTelemetry`)
- Components receive `Snapshot` prop — no business logic inside
- Tailwind utility classes, no custom CSS
- Double quotes for strings with apostrophes

### Git Commits
- Update documentation in same commit as code changes
- Run `make test` before commit
- Update `README.md` "Last reviewed against code" date

---

## Debugging Checklist for Common Issues

| Symptom | Check |
|---------|-------|
| No camera frames | macOS permission, `--list-cameras`, another app using camera |
| Low FPS | `--imgsz`, `display_scale`, `--fps`, camera hardware limit |
| No detections | Model path, confidence threshold, `--device`, class filter (COCO 0) |
| Tracks not stable | `max_age`, `min_hits`, `iou_threshold` in SortTracker |
| ROI not counting | `config/camera.yaml` roi_configured, polygon area > 0 |
| Line not counting | `counting_line` configured, `xray_side_point` set |
| State stuck | `hysteresis`, `debounce_frames`, `congested_require_spacing` |
| API 404 | Vite proxy running, backend on 8080, CORS origins |
| MJPEG not loading | `/api/camera/stream` endpoint, holder has frame, CORS |
| Dashboard shows "OFFLINE" | Backend running with `--api`, source selector = "Live API" |