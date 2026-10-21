# Validation Methodology

## Purpose

Ensure that future changes do not silently break queue counting accuracy. This document defines automated tests, manual validation scenarios, and synchronization checks between backend and dashboard.

---

## Validation Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION PYRAMID                                   │
└─────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────┐
        │  CONTROLLED     │  ← Manual scenarios with known ground truth
        │  MANUAL TESTS   │     (0 people → EMPTY, 5 people → NORMAL, etc.)
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  OFFLINE        │  ← Evaluation against labeled footage
        │  EVALUATION     │     (scripts/evaluate_counts.py)
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  INTEGRATION    │  ← Dashboard ↔ API ↔ Backend sync
        │  TESTS          │     (live camera validation)
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  UNIT TESTS     │  ← 137 tests (synthetic tracks, no camera)
        │  (137 PASSING)  │     Run: make test
        └─────────────────┘
```

---

## 1. Unit Tests (Automated)

**Command:** `make test` or `venv/bin/python -m unittest discover -s tests`
(297 tests at the time of the mapping feature; `venv/bin/python -m pytest
tests/ -q` = 350 for the pytest target)

| Test Module | Coverage |
|-------------|----------|
| `test_queue_prototype.py` | Camera-only analytics (occupancy, flow, waiting, spacing, classifier, stopwatch, events, ROI, calibration, dashboard rendering) + mapping-driven crossings: IN/OUT semantics, occupancy-is-membership regression (IN crossing must NOT finalise the exit), debounce/dedup/jitter, reversed crossings, bottom_center vs centroid reference, debounce knob |
| `test_mapping.py` | Mapping schema: normalised↔pixel round trips, full validation matrix (bounds, vertex count, degenerate area, self-intersection, short lines, identical/degenerate lines, direction/reference/schema checks), legacy conversion for horizontal AND vertical counting lines (X-ray-side orientation), atomic save + backup + reload |
| `test_mapping_api.py` | `GET/PUT /api/dashboard/mapping` against a temp config: legacy derivation reported, list AND dict line forms, backup created on update, invalid payload rejected with the last known-good file untouched, config endpoint additive fields |
| `test_evaluate_counts.py` | Offline evaluator (alignment, MAE, ±1/±2 accuracy, bias, gaps) |
| `test_backlog_alerts.py` | Legacy alert engine |
| `test_fusion.py` | Legacy sensor fusion |
| `test_metrics.py` | Legacy metrics |
| `test_monitor.py` | Legacy monitor integration |
| `test_state_machine.py` | Legacy state machine |

**Key Properties:**
- Synthetic tracks only — no camera/YOLO required
- Fast (< 1 second)
- Deterministic
- Run on every commit

---

## 2. Smoke Test (Automated)

**Command:** `make smoke`

**Runs:** `venv/bin/python app/detect_people.py --source output/smoke_test/bus.jpg`

**Expected Output:**
```
[INFO] mapping 'CAMERA 01' active from mappings.yaml — roi=4pt, lines=['in', 'out'], ref=bottom_center
...
PERSONS: 3   STATE: UNCONFIGURED   (image mode — no FPS)
```

**Validates:**
- YOLO model loads
- Inference runs
- Pipeline processes single frame
- Camera mappings.yaml loads and validates (falls back to legacy
  geometry with a `[WARN]` if the file is broken)
- No crashes on image input

**Note:** image sources stop after a single frame, so the API stream and
the `PUT`-driven hot reload are only exercised by a video/camera source
(see §5).

---

## 3. Offline Evaluation (Quantitative)

**Tool:** `scripts/evaluate_counts.py`

### Workflow

```bash
# 1. Record 1 Hz occupancy samples from video
./run.sh --source input/airport_cctv.mp4 --record-samples output/samples.jsonl

# 2. Create manual ground truth CSV
# Format: start_s,end_s,count[,note]
# Example:
# start_s,end_s,count,note
# 0.0,10.0,7,
# 10.0,20.0,9,two people left

# 3. Evaluate
venv/bin/python scripts/evaluate_counts.py \
    --samples output/samples.jsonl \
    --ground-truth output/ground_truth.csv \
    --tolerance-s 1.0 \
    --gap-s 10.0 \
    --min-coverage 0.8 \
    --json output/eval_report.json
```

### Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **MAE** | Mean absolute error | < 1.0 |
| **±1 Accuracy** | Fraction with \|error\| ≤ 1 | > 0.85 |
| **±2 Accuracy** | Fraction with \|error\| ≤ 2 | > 0.95 |
| **Bias** | Signed mean error | \|Bias\| < 0.5 |

### Alignment Rules

- Each ground-truth interval matched to prediction sample nearest its midpoint
- Match must lie within `[start_s - tolerance_s, end_s + tolerance_s]`
- Recording gaps detected and flagged; intervals in gaps excluded and reported
- If coverage < `--min-coverage`, no accuracy verdict emitted

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Insufficient data |
| 2 | Malformed input |

---

## 4. Controlled Manual Validation Scenarios

**Purpose:** Verify qualitative behavior with known person counts.

### Test Setup
- Use `--define-roi` to configure a test ROI
- Use `--api` to enable dashboard
- Position known number of people in queue area
- Observe state, occupancy, flow, waiting

### Scenarios

| Scenario | People in ROI | Expected State | Expected Occupancy | Notes |
|----------|---------------|----------------|-------------------|-------|
| **Empty** | 0 | `EMPTY` | 0% | No tracks in ROI |
| **Sparse** | 1-2 | `SPARSE` | 3-7% (cap=30) | Occupancy ≥ 10% |
| **Normal** | 3-5 | `NORMAL` | 10-20% | Occupancy ≥ 40% |
| **Crowded** | 6-8 | `CROWDED` | 20-27% | Occupancy ≥ 60% |
| **Congested** | 9-12 | `CONGESTED` | 30-40% | Occupancy ≥ 80% + spacing ≤ 60px |
| **Backlog** | 15+ | `BACKLOG` | ≥ 60% | Requires baseline + sustained |

> **Thresholds from `config/camera.yaml` (default):**
> - `occupancy_sparse: 0.10` (3 people at cap=30)
> - `occupancy_normal: 0.40` (12 people)
> - `occupancy_crowded: 0.60` (18 people)
> - `occupancy_congested: 0.80` (24 people)
> - `congested_max_spacing_px: 60`
> - `congested_require_spacing: true`

### Validation Checklist per Scenario

| Check | How to Verify |
|-------|---------------|
| People count | Console: `IN-QUEUE: N` matches visible count |
| State | Dashboard chip + console `STATE: X` |
| Occupancy % | Dashboard meter + console `OCCUPANCY: XX%` |
| Flow | ENTER/EXIT events on crossing |
| Waiting time | Increases while in queue, resets on EXIT |
| Spacing | Decreases as people get closer |
| Congestion timer | Starts on CONGESTED, shows MM:SS |

---

## 5. Integration Tests (Live Camera)

### Backend → API Sync

```bash
# Terminal 1: Start backend with API
./run.sh --api --source 0

# Terminal 2: Verify API
curl -s http://localhost:8080/api/snapshot | python -m json.tool
# Check: people_in_queue matches OpenCV window
```

### API → Dashboard Sync

```bash
# Terminal 3: Start dashboard
cd Queue-monitoring-dashboard && npm run dev

# Browser: http://localhost:8443
# Select "Live API" from Source dropdown
```

**Sync Checks:**

| Check | Backend (OpenCV) | Dashboard (React) | Must Match |
|-------|------------------|-------------------|------------|
| People in queue | Console `IN-QUEUE` | QueueStatusPanel big number | ✅ |
| State | Console `STATE` | QueueStatusPanel chip | ✅ |
| Occupancy % | Console `OCCUPANCY` | Occupancy meter | ✅ |
| Flow (entry/exit) | Console `ENTR/EXIT` | KpiStack tiles | ✅ |
| Waiting | Console `AVG WAIT` | KpiStack Avg Wait | ✅ |
| Events | Console `REASONS` | EventTimeline | ✅ |
| Congestion timer | Console `cong:` | QueueStatusPanel strip | ✅ |
| Tracks on video | OpenCV boxes | CameraPanel overlays | ✅ |
| MJPEG stream | — | CameraPanel video | Loads |

### MJPEG Stream Validation

```bash
# Direct stream test (annotated operator view)
curl -N --max-time 5 http://localhost:8080/api/camera/stream | head -5
# Should show MJPEG boundary + JPEG data

# Raw variant (mapping editor view — same bytes without burned-in geometry)
curl -N --max-time 5 "http://localhost:8080/api/camera/stream?annotated=0" | head -5

# Browser test
# <img src="http://localhost:8080/api/camera/stream" />
# Should display live video with overlays
```

---

## 5b. Mapping & Calibration Validation

Geometry is resolution-independent (normalised coordinates), so the desk AND
the live-energy deployment validate the same assertions:

1. **Geometry source rule** — with a valid `config/mappings.yaml` entry, the
   backend logs `[INFO] mapping 'CAMERA 01' active from mappings.yaml — …`.
   Delete/corrupt the file and it must fall back to legacy geometry with a
   `[WARN]` (never a silent mix of the two).
2. **Crossing semantics regression** — covered by
   `tests/test_queue_prototype.py::TestMappingDrivenCrossings`: forward IN
   registers an entry (occupancy stays ROI membership, exit state untouched),
   forward OUT finalises exit + waiting time, reversed crossings are explicit
   events that change no counter, jitter never fires, and
   `crossing_debounce_frames` dedups multi-frame side changes.
3. **Calibration round trip in the UI** (API + dashboard running):

   | Step | Expected |
   |------|----------|
   | Open dashboard → "Edit mapping" | Editor over the raw (`?annotated=0`) feed, letterboxed so every handle is reachable, saved geometry ghosted |
   | Drag a handle / nudge with arrow keys | `Unsaved changes` chip; Revert/Save/Apply enable |
   | Save | `Mapping applied…`; `mappings.yaml` rewritten atomically + `.bak`; backend logs `Config reloaded via api` on the next frame; per-track crossing state reset |
   | Send a deliberately invalid mapping (e.g. degenerate ROI) | `applied: false` with errors, disk untouched, running mapping still active |
   | Close with Cancel | Stream returns to the annotated operator view, monitoring intact |

4. **Reference point** — `bottom_center` (calibrated default) scores the
   person at the bbox bottom-edge middle; switching `reference_point` to
   `centroid` (`camera.yaml` override or mappings.yaml value) is a pure
   analytics toggle verified by unit tests, no geometry change.

---

## 6. Dashboard Validation

### Source Selector
- **Simulator** → Shows synthetic data (NORMAL/BUSY/CONGESTED/BACKLOG scenarios)
- **Live API** → Shows real backend data (requires `./run.sh --api`)

### Component Checks

| Component | Verify |
|-----------|--------|
| QueueStatusPanel | State chip, occupancy meter, trend/wait/congestion strip |
| CameraPanel | MJPEG loads, overlays (boxes, IDs, ROI, line) toggleable |
| KpiStack | 9 tiles with sparklines, correct colors |
| AnalyticsPanel | FlowChart + OccupancyChart, range selector works |
| EventTimeline | Filter (All/Flow/Alerts) works, events appear |
| DiagnosticsDrawer | Opens on health chip click, shows raw snapshot |

---

## 7. Regression Test Checklist (Pre-Commit)

### Automated
- [ ] `make test` — all 137 tests pass
- [ ] `make smoke` — offline image test passes

### Manual (with camera or video)
- [ ] Camera opens, `● LIVE` indicator increments
- [ ] YOLO detects people (boxes appear)
- [ ] SORT assigns stable IDs (same person = same ID)
- [ ] ROI works (outside = grey, inside = colored)
- [ ] Counting line: ENTER/EXIT events fire correctly
- [ ] Occupancy % = people_in_queue / queue_capacity
- [ ] Flow rates update per minute
- [ ] State classification: EMPTY → SPARSE → NORMAL → CROWDED → CONGESTED
- [ ] Congestion stopwatch: starts/stops correctly
- [ ] Events logged: `--record` produces valid JSONL
- [ ] API `/api/snapshot` returns valid JSON
- [ ] API `/api/camera/stream` serves MJPEG
- [ ] Dashboard "Live API" shows MJPEG + overlays
- [ ] Dashboard "Simulator" shows synthetic data
- [ ] Source selector switches cleanly

---

## 8. Evaluation Gates for Deployment

Before claiming any accuracy:

| Gate | Requirement |
|------|-------------|
| **Representative Footage** | Real airport CCTV, same perspective as deployment |
| **Ground Truth** | Manual annotation at midpoint of intervals |
| **Sample Size** | ≥ 30 minutes footage, ≥ 50 ground-truth intervals |
| **MAE** | < 1.0 people |
| **±1 Accuracy** | > 85% |
| **±2 Accuracy** | > 95% |
| **Bias** | \|Bias\| < 0.5 |
| **Coverage** | ≥ 80% (no large recording gaps) |

> **Never claim "deployment-grade" or "X% accurate" without passing all gates.**

---

## 9. Validation Tools Reference

| Tool | Command | Purpose |
|------|---------|---------|
| Unit tests | `make test` | Synthetic track validation |
| Smoke test | `make smoke` | Quick pipeline check |
| Camera test | `./run.sh --test-camera` | Camera FPS + frame verification |
| List cameras | `./run.sh --list-cameras` | Find correct camera index |
| ROI preview | `./run.sh --roi-preview` | Verify saved layout |
| Define ROI | `./run.sh --define-roi` | Interactive layout setup |
| Record events | `./run.sh --record out.jsonl` | Event logging |
| Record samples | `./run.sh --record-samples out.jsonl` | 1 Hz occupancy for evaluation |
| Evaluate | `scripts/evaluate_counts.py` | Quantitative accuracy |
| API health | `curl /api/health` | Pipeline status |
| API snapshot | `curl /api/snapshot` | Real-time data |
| Dashboard | `npm run dev` | UI validation |

---

## 10. Continuous Validation

### On Every Code Change
1. Run `make test`
2. If camera-related: test with `./run.sh --test-camera` + live camera
3. If API-related: verify `curl /api/snapshot` + dashboard sync
4. If classifier-related: run offline evaluation with test footage

### Before Release/Deploy
1. Full manual validation suite (Section 4)
2. Offline evaluation on representative footage (Section 3)
3. Dashboard/API sync with live camera (Section 5)
4. Update `README.md` "Last reviewed against code" date