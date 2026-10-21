# Configuration Reference

## Overview

All runtime configuration for the camera-only prototype lives in **`config/camera.yaml`**.

This file is the **single source of truth** for:
- Camera capture settings
- Queue layout (ROI, counting line, X-ray side)
- Classifier thresholds
- Backlog conditions
- Calibration
- Dashboard layout
- Movement classification

---

## Complete Schema with Status Markers

### IMPLEMENTED vs PLANNED Legend

| Marker | Meaning |
|--------|---------|
| ✅ **IMPLEMENTED** | Actively used in current runtime |
| ⚠️ **DISABLED** | Code exists but disabled by default (requires config) |
| 🔮 **PLANNED** | Placeholder for future phase, not used |
| 📝 **DOCUMENTED** | Documented in schema but may need calibration |

---

```yaml
# Airport X-Ray Screening Queue — Camera-Only Prototype configuration.
# All values are camera-relative (pixel space). No real-world (metre) value
# is assumed until calibration is supplied. Thresholds are EXAMPLE values
# to be calibrated against real CCTV footage later.

camera:                          # ✅ IMPLEMENTED
  # Operator-facing camera label shown in dashboard header
  camera_id: "CAMERA 01"         # ✅ IMPLEMENTED

  # Camera source: webcam index ("0", "1", ...), video file path,
  # image path, or RTSP/HTTP URL. CLI --source overrides this.
  source: 0                      # ✅ IMPLEMENTED

  # Capture resolution requested from camera. 0 = native (recommended).
  # If unsupported, camera falls back to nearest supported resolution.
  width: 0                       # ✅ IMPLEMENTED
  height: 0                      # ✅ IMPLEMENTED

  # Display scale relative to captured frame width (1.0 = full native).
  # 0.5 = half width. Lower = faster rendering, smaller window.
  display_scale: 1.0             # ✅ IMPLEMENTED

  # Detection target: "person" (full body, default) or "head" (head-crop).
  # "head" mode for steep camera angles where bodies are occluded.
  detect_mode: "person"          # ✅ IMPLEMENTED

  # Queue ROI polygon, pixel coordinates [[x, y], ...].
  # Leave null and run `python app/detect_people.py --define-roi`
  # to draw it with the webcam, or edit directly.
  roi: null                      # ✅ IMPLEMENTED (null = unconfigured)

  # Virtual counting line, two endpoints [[x1, y1], [x2, y2]].
  counting_line: null            # ✅ IMPLEMENTED (null = unconfigured)

  # Reference pixel on the X-RAY side of the counting line.
  #   Crossing toward this point = EXIT (processed / throughput)
  #   Crossing away from it      = ENTER (joining queue)
  xray_side_point: null          # ✅ IMPLEMENTED (null = unconfigured)

  # Expected maximum people in ROI → normalized occupancy %.
  queue_capacity: 30             # ✅ IMPLEMENTED

  # Flow windows in minutes. First = "current" window, rest = rolling averages.
  flow_windows_minutes: [1, 5]   # ✅ IMPLEMENTED

  # How long a vanished track is kept as "in queue" before dropped.
  track_lost_grace_seconds: 1.5  # ✅ IMPLEMENTED

classifier:                      # ✅ IMPLEMENTED
  # Occupancy-ratio ladder for EMPTY / SPARSE / NORMAL / CROWDED / CONGESTED.
  # Ratios are people_in_queue / queue_capacity.
  occupancy_sparse: 0.10         # 📝 EXAMPLE — calibrate
  occupancy_normal: 0.40         # 📝 EXAMPLE — calibrate
  occupancy_crowded: 0.60        # 📝 EXAMPLE — calibrate
  occupancy_congested: 0.80      # 📝 EXAMPLE — calibrate

  # Slack before classifier downgrades (anti-flapping).
  # Downgrade threshold = threshold × (1 - hysteresis)
  hysteresis: 0.05               # 📝 EXAMPLE — calibrate

  # Consecutive frames with same candidate before state commits.
  debounce_frames: 5             # 📝 EXAMPLE — calibrate

  # CONGESTED requires mean pixel spacing ≤ this value.
  congested_max_spacing_px: 60.0 # 📝 EXAMPLE — calibrate
  congested_require_spacing: true # 📝 EXAMPLE — calibrate

backlog:                         # ⚠️ DISABLED (baseline = null)
  # BACKLOG fires only when ALL conditions hold AND sustained:
  min_people: 15                 # 📝 EXAMPLE — calibrate
  min_occupancy: 0.60            # 📝 EXAMPLE — calibrate
  min_wait_seconds: 300.0        # 📝 EXAMPLE — calibrate
  persistence_seconds: 10.0      # 📝 EXAMPLE — calibrate
  flow_ratio_below_baseline: 0.5 # 📝 EXAMPLE — calibrate

  flow:
    # Expected normal throughput (people/min), measured from REAL footage.
    # null → BACKLOG classification disabled. Reports:
    # "BACKLOG CHECK UNAVAILABLE — FLOW BASELINE NOT CONFIGURED"
    baseline_people_per_minute: null  # 🔧 MUST SET TO ENABLE

calibration:                     # 🔮 PLANNED (not used without data)
  # Fill in only when camera calibration / homography is available.
  # Until then all measurements stay pixel-relative (uncalibrated).
  homography_path: null          # 🔮 PLANNED
  px_per_metre: null             # 🔮 PLANNED

source:                          # ✅ IMPLEMENTED (recorded by --define-roi)
  shape_w: null                  # Frame width when ROI was defined
  shape_h: null                  # Frame height when ROI was defined

# Queue-movement classification for dashboard: MOVING / STAGNANT / IDLE
movement:                        # ✅ IMPLEMENTED
  window_s: 60                   # Rolling window (seconds) — 📝 EXAMPLE
  min_events: 1                  # Line crossings in window to count as MOVING

dashboard:                       # ✅ IMPLEMENTED
  # Rolling average window (seconds) for smoothed in-queue people in charts.
  # Display only — NEVER feeds classifier.
  occupancy_avg_window_s: 120    # 📝 EXAMPLE

  # Right status panel width (px at UI scale 1.0)
  panel_width: 360               # 📝 EXAMPLE

  # Samples kept per live trend chart
  chart_history: 120             # 📝 EXAMPLE

# Per-state operational recommendations. Override any/all states here.
# Unlisted states use built-in defaults.
# recommendations:               # ✅ IMPLEMENTED
#   CONGESTED: "Congestion detected. Open secondary lanes immediately."
#   BACKLOG: "Activate queue-management protocol."
```

---

## Field-by-Field Details

### Camera Section

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `camera_id` | string | `"CAMERA 01"` | No | Label in dashboard header |
| `source` | string\|int | `0` | No | Webcam index, video path, or RTSP/HTTP URL. CLI `--source` overrides |
| `width` | int | `0` | No | Requested capture width (0 = native) |
| `height` | int | `0` | No | Requested capture height (0 = native) |
| `display_scale` | float | `1.0` | No | Window size relative to capture (0.5 = half) |
| `detect_mode` | string | `"person"` | No | `"person"` or `"head"` |
| `roi` | array\|null | `null` | **Yes** for analytics | Polygon `[[x,y],...]` in pixels |
| `counting_line` | array\|null | `null` | **Yes** for flow | Two endpoints `[[x1,y1],[x2,y2]]` |
| `xray_side_point` | array\|null | `null` | **Yes** for direction | Reference pixel `[x,y]` on X-ray side |
| `queue_capacity` | int | `30` | No | Max expected people → occupancy % |
| `flow_windows_minutes` | array | `[1,5]` | No | First = current, rest = rolling avg |
| `track_lost_grace_seconds` | float | `1.5` | No | Track kept after loss |

### Classifier Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `occupancy_sparse` | float | `0.10` | Ratio → SPARSE |
| `occupancy_normal` | float | `0.40` | Ratio → NORMAL |
| `occupancy_crowded` | float | `0.60` | Ratio → CROWDED |
| `occupancy_congested` | float | `0.80` | Ratio → CONGESTED |
| `hysteresis` | float | `0.05` | Downgrade slack (anti-flapping) |
| `debounce_frames` | int | `5` | Frames before state commits |
| `congested_max_spacing_px` | float | `60.0` | CONGESTED requires spacing ≤ this |
| `congested_require_spacing` | bool | `true` | If false, spacing ignored for CONGESTED |

**State Transitions (with hysteresis):**

```
EMPTY (occ < 0.10×0.95)
    ↓ occ ≥ 0.10
SPARSE (occ < 0.40×0.95)
    ↓ occ ≥ 0.40
NORMAL (occ < 0.60×0.95)
    ↓ occ ≥ 0.60
CROWDED (occ < 0.80×0.95)
    ↓ occ ≥ 0.80 + spacing ≤ 60px
CONGESTED
    ↓ (all backlog conditions + persistence)
BACKLOG
```

### Backlog Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_people` | int | `15` | Minimum people in queue |
| `min_occupancy` | float | `0.60` | Minimum occupancy ratio |
| `min_wait_seconds` | float | `300` | Minimum mean wait |
| `persistence_seconds` | float | `10.0` | Seconds all conditions must hold |
| `flow_ratio_below_baseline` | float | `0.5` | Current flow < baseline × this |
| `baseline_people_per_minute` | float\|null | `null` | **Must set to enable BACKLOG** |

**BACKLOG Logic (ALL must be true):**
```
people_in_queue ≥ min_people
AND occupancy_ratio ≥ min_occupancy
AND flow_current < baseline × flow_ratio_below_baseline
AND mean_wait_s ≥ min_wait_seconds
AND sustained for persistence_seconds
```

> **Critical:** If `baseline_people_per_minute` is `null` (default), BACKLOG is **disabled** and snapshot reports:
> `"BACKLOG CHECK UNAVAILABLE — FLOW BASELINE NOT CONFIGURED"`

### Calibration Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `homography_path` | string\|null | `null` | Path to homography matrix file |
| `px_per_metre` | float\|null | `null` | Pixels per metre (if no homography) |

- **Both null** → All measurements pixel-relative, HUD shows `CALIBRATION: PX ONLY`
- **Either set** → Physical distances enabled, HUD shows `CALIBRATION: METRES`

### Movement Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_s` | float | `60` | Rolling window for line crossings |
| `min_events` | int | `1` | Crossings in window → MOVING |

**States:**
- `IDLE` — Queue empty
- `STAGNANT` — People present, no crossings in window
- `MOVING` — ≥ `min_events` crossings in `window_s`

### Dashboard Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `occupancy_avg_window_s` | float | `120` | Chart-only rolling avg (display only) |
| `panel_width` | int | `360` | Right panel width (px at scale 1.0) |
| `chart_history` | int | `120` | Samples per trend chart |

### Recommendations Section

```yaml
recommendations:
  CONGESTED: "Custom message for operators"
  BACKLOG: "Custom backlog protocol text"
```

Overrides built-in defaults for specific states.

---

## Configuration Workflow

### 1. Initial Setup (Required)
```bash
./run.sh --define-roi
```
Interactive tool draws ROI, counting line, X-ray side point → saves to `camera.yaml`.

### 2. Verify Layout
```bash
./run.sh --roi-preview
```

### 3. Calibrate Classifier Thresholds
Edit `config/camera.yaml` based on evaluation results:
```bash
# After running scripts/evaluate_counts.py with real footage
# Adjust occupancy_* thresholds to match observed states
```

### 4. Enable BACKLOG (When Ready)
```yaml
backlog:
  flow:
    baseline_people_per_minute: 12.5  # Measured from real footage
```

### 5. Add Calibration (When Available)
```yaml
calibration:
  homography_path: "calibration/homography.npy"
  # OR
  px_per_metre: 45.2
```

---

## Validation Checklist

Before deploying to a new camera:

- [ ] `camera_id` set to meaningful label
- [ ] `source` correct (webcam index / video path / RTSP URL)
- [ ] `width`/`height` = 0 (native) or supported resolution
- [ ] `display_scale` appropriate (1.0 = full, 0.5 = half for performance)
- [ ] `detect_mode` = `"person"` (or `"head"` for steep angles)
- [ ] ROI drawn with `--define-roi` (≥3 points, area > 0)
- [ ] Counting line drawn (2 endpoints across queue flow)
- [ ] `xray_side_point` on X-ray side of line
- [ ] `queue_capacity` matches physical queue capacity
- [ ] Classifier thresholds reviewed (default = example values)
- [ ] `hysteresis` and `debounce_frames` prevent flapping
- [ ] `congested_max_spacing_px` matches camera perspective
- [ ] `backlog.flow.baseline_people_per_minute` set (or leave null)
- [ ] `calibration` = null (pixel-only) or populated
- [ ] `movement.window_s` / `min_events` match observed flow
- [ ] Dashboard `panel_width` / `chart_history` appropriate

---

## CLI Overrides

| Config Field | CLI Flag | Precedence |
|--------------|----------|------------|
| `camera.source` | `--source` | CLI > Config |
| `camera.detect_mode` | `--detect-mode` | CLI > Config |
| `camera.width`/`height` | (not exposed) | Config only |
| Config file path | `--camera-config` | CLI > Default |

---

## Environment-Specific Configs

For multiple cameras/checkpoints, create separate configs:
```bash
# Checkpoint A
./run.sh --camera-config config/camera_checkpoint_A.yaml --api --api-port 8080

# Checkpoint B
./run.sh --camera-config config/camera_checkpoint_B.yaml --api --api-port 8081
```

Each runs independent pipeline with own ROI, thresholds, API port.