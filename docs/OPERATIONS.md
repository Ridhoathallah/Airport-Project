# Operations Guide

## Prerequisites

### System Requirements
- **OS:** macOS (Apple Silicon recommended) or Linux
- **Python:** 3.9 (developed on 3.9.6)
- **Node.js:** 18+ (for dashboard)
- **pnpm:** 8+ (or npm)
- **Camera:** USB webcam, IP camera (RTSP/HTTP), or video files for offline testing

### Hardware
- Webcam (tested with 1920×1080 @ 30 FPS)
- For Apple Silicon: MPS acceleration used automatically
- Minimum 8 GB RAM recommended

---

## Installation

### Backend (airport_people_counter)

```bash
cd /path/to/Airport-Project/airport_people_counter

# Option 1: Use the launcher (recommended — auto-creates venv + installs deps)
./run.sh --help

# Option 2: Manual setup
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements.txt

# Verify
python -c "import cv2, torch, numpy, yaml; print('cv2', cv2.__version__); print('torch', torch.__version__)"
```

### Dashboard (Queue-monitoring-dashboard)

```bash
cd /path/to/Airport-Project/Queue-monitoring-dashboard

# Install dependencies
npm install
# or
pnpm install

# Verify build
npm run build
```

---

## Camera Setup

### Discover Available Cameras

```bash
# List cameras with indices, resolution, FPS, and macOS device names
./run.sh --list-cameras
# or
venv/bin/python app/detect_people.py --list-cameras
```

Output example:
```
[index]  resolution  reported_fps  measured_fps  device_name
0        1920x1080   30.0          29.8          "1080P Web Camera"
1        1280x720    30.0          29.9          "FaceTime HD Camera"
```

### Test a Camera (without YOLO)

```bash
# Test default camera (index 0)
./run.sh --test-camera

# Test specific camera
./run.sh --test-camera 1
```

This opens the camera, measures actual FPS, and verifies frames are not frozen.

### macOS Camera Permissions

**Required:** Grant camera access to your terminal application.

1. System Settings → Privacy & Security → Camera
2. Enable your terminal (Terminal.app, iTerm2, VS Code, etc.)
3. **Relaunch the terminal** after granting permission

If the app starts but exits immediately with "delivered no frames", this is almost always a permission issue.

---

## Running the System

### Quick Start (Two Terminals)

#### Terminal 1 — Camera + API Backend
```bash
cd airport_people_counter

# Webcam + API (port 8080)
./run.sh --api

# Or with specific camera
./run.sh --api --source 1

# Or with video file + API
./run.sh --api --source input/video.mp4
```

#### Terminal 2 — React Dashboard
```bash
cd Queue-monitoring-dashboard
npm run dev
# Vite starts on http://localhost:8443
```

#### Browser
Open **http://localhost:8443** → Select **"Live API"** from Source dropdown in top bar.

---

## Common Run Commands

### Backend (via `./run.sh` or `make`)

| Command | Description |
|---------|-------------|
| `./run.sh` | Webcam, default config |
| `./run.sh --api` | Webcam + FastAPI server (port 8080) |
| `./run.sh --define-roi` | Interactive ROI + counting line setup |
| `./run.sh --roi-preview` | Preview saved layout on live frame |
| `./run.sh --source input/video.mp4` | Run on video file |
| `./run.sh --source http://ip:port/video` | IP camera (RTSP/HTTP) |
| `./run.sh --record output/events.jsonl` | Record analytics events (JSONL) |
| `./run.sh --record-samples output/samples.jsonl` | Record 1 Hz occupancy samples |
| `./run.sh --debug` | Show classifier reasons on HUD |
| `./run.sh --legacy-hud` | Use legacy full-frame HUD |
| `./run.sh --imgsz 320` | Lower inference size for speed |
| `./run.sh --fps 15` | Cap at 15 FPS |
| `./run.sh --device cpu` | Force CPU (disable MPS) |

### Makefile Shortcuts
```bash
make run           # ./run.sh
make api           # ./run.sh --api
make define-roi    # ./run.sh --define-roi
make preview       # ./run.sh --roi-preview
make record        # ./run.sh --record output/events.jsonl
make test          # Run unit tests
make smoke         # Quick test on sample image
make test-camera   # Test camera (default index 0)
make list-cameras  # List cameras
```

### Dashboard
```bash
cd Queue-monitoring-dashboard
npm run dev        # Dev server on :8443
npm run build      # Production build to dist/
npm run preview    # Preview production build
npm run format     # Format with oxfmt
```

---

## ROI Configuration (Required for Analytics)

The **ROI (Region of Interest)** defines the queue area. Only people whose centroid falls inside the ROI count toward occupancy.

### Interactive Setup
```bash
./run.sh --define-roi
```

**Steps:**
1. Camera opens showing live frame
2. **STEP 1/3 — Draw ROI polygon:** Click ≥3 points around queue area. Press **Enter** to finish.
3. **STEP 2/3 — Define counting line:** Click 2 endpoints across where people walk toward X-ray. Press **Enter**.
4. **STEP 3/3 — Define X-ray side:** Click 1 reference point on the X-ray/processing side. Press **Enter**.
5. Layout validated and saved to `config/camera.yaml`.

**Keys during drawing:** `r` = reset step, `b` = back a step, `s` = save immediately, `Esc` = cancel.

### Verify Layout
```bash
./run.sh --roi-preview
```

### Manual Edit
Edit `config/camera.yaml` directly:
```yaml
camera:
  roi: [[x1, y1], [x2, y2], [x3, y3], ...]
  counting_line: [[x1, y1], [x2, y2]]
  xray_side_point: [x, y]
```

---

## Counting Line Direction

The counting line direction is **fully configurable** — no fixed camera orientation assumed.

- **ENTER** = track crosses line **away from** `xray_side_point` (joining queue)
- **EXIT** = track crosses line **toward** `xray_side_point` (throughput/processed)

The `xray_side_point` is a single pixel reference on the X-ray side. The cross-product sign selects the "toward/away" half-plane.

---

## Configuration Reference

All runtime configuration lives in **`config/camera.yaml`**. See [docs/CONFIGURATION.md](CONFIGURATION.md) for complete schema.

Key sections:
- `camera:` — capture, display, detection mode, ROI, line, capacity
- `classifier:` — occupancy thresholds, hysteresis, debounce, spacing
- `backlog:` — BACKLOG conditions (disabled by default)
- `calibration:` — homography / px_per_metre (null = pixel-only)
- `movement:` — MOVING/STAGNANT/IDLE classification
- `dashboard:` — panel width, chart history, occupancy avg window
- `recommendations:` — per-state operational text overrides

---

## Ports

| Service | Port | Configurable |
|---------|------|--------------|
| FastAPI Backend | 8080 | `--api-port` |
| Vite Dev Server | 8443 | `PORT` env var |
| Vite Proxy | — | `/api/*` → `localhost:8080` |

---

## Recording & Evaluation

### Record Events (JSONL)
```bash
./run.sh --api --record output/events.jsonl
```

### Record Occupancy Samples (1 Hz, for evaluation)
```bash
./run.sh --api --record-samples output/samples.jsonl
```

### Evaluate Against Ground Truth
```bash
# 1. Record samples
./run.sh --source input/cctv.mp4 --record-samples output/samples.jsonl

# 2. Create ground truth CSV (start_s,end_s,count[,note])
# Example:
# start_s,end_s,count,note
# 0.0,10.0,7,
# 10.0,20.0,9,two people left

# 3. Run evaluator
venv/bin/python scripts/evaluate_counts.py \
    --samples output/samples.jsonl \
    --ground-truth output/ground_truth.csv \
    --json output/eval_report.json
```

Metrics: MAE, ±1 accuracy, ±2 accuracy, Bias.

---

## Troubleshooting

### Camera Not Opening
- Check `--source` value
- Verify camera not in use by another app
- Check macOS camera permission (see above)
- Run `--list-cameras` to find correct index

### Wrong Camera Device
```bash
./run.sh --list-cameras
./run.sh --source <index>
```

### Low FPS
- Lower inference size: `--imgsz 320`
- Lower display scale: `camera.display_scale: 0.5` in `camera.yaml`
- Lower target FPS: `--fps 15`
- Close other apps using camera
- Note: Some USB webcams only deliver ~15 FPS; the loop paces to camera

### Frozen OpenCV Window (macOS)
- The code mitigates this with explicit `cv2.namedWindow(WINDOW_NORMAL)` and `cv2.waitKey(1)` every frame
- Try `camera.display_scale: 0.5`
- Check `● LIVE` indicator and `FRM` counter in dashboard — if incrementing, pipeline is alive

### ROI Not Configured
HUD shows `ROI NOT CONFIGURED — run --define-roi`

### Counting Line Not Configured
Re-run `--define-roi` and complete STEP 2/3, or edit `counting_line` / `xray_side_point` in `camera.yaml`

### BACKLOG Unavailable
Set `backlog.flow.baseline_people_per_minute` in `camera.yaml` to a measured value from real footage.

### Unsupported Resolution
If requested `width/height` unsupported, camera falls back to nearest supported size (logged at startup).

---

## Testing

### Unit Tests
```bash
cd airport_people_counter
make test
# or
venv/bin/python -m unittest discover -s tests
# or
venv/bin/python -m pytest tests -q
```

### Targeted Test Runs
```bash
# Camera-only analytics (no camera/YOLO)
venv/bin/python -m unittest tests.test_queue_prototype -v

# Offline evaluator
venv/bin/python -m unittest discover -s tests -p "test_evaluate_counts.py"
```

### Smoke Test (Offline)
```bash
make smoke
# Runs on output/smoke_test/bus.jpg (single frame, no FPS)
```

---

## Logs & Output

- **Console:** Live metrics line (DET, IN-QUEUE, ENTR, EXIT, FPS, timings, STATE, MOV, REASONS)
- **OpenCV Window:** Monitoring dashboard (default) or legacy HUD (`--legacy-hud`)
- **Recorded Events:** `--record output/events.jsonl` (JSON Lines)
- **Occupancy Samples:** `--record-samples output/samples.jsonl` (1 Hz JSONL)
- **Screenshots:** `output/` directory