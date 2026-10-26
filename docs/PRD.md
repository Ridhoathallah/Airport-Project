# PRD — Airport X-Ray Queue Monitoring System

**Status:** living document, aligned with the repository as of 2026-10-21.
Every status marker below was verified against code and tests in this repository
(`Implemented` = evidence exists; `Partial` = code exists but incomplete/disabled;
`Planned` = not built; `Unknown` = cannot be confirmed from this repository).

---

## 1. Product Summary

A **camera-only queue analytics prototype** for an airport X-ray security
checkpoint. It detects and tracks people in a webcam/video/RTSP feed (YOLOv5s +
SORT), measures queue behaviour inside a configurable area (occupancy, entry/exit
flow, waiting time, congestion state), and shows it all on a real-time operator
dashboard. Everything runs locally on one machine — no cloud, no identity
recognition.

- Backend source of truth: `airport_people_counter/` (Python 3.9, FastAPI)
- Presentation only: `Queue-monitoring-dashboard/` (React 19, Vite 8, Tailwind v4)
- One-command launcher: `./start.sh`

## 2. Background & Problem

Airport screening lanes suffer from unpredictable waits. Staff currently judge
queue conditions by eye, which gives no early warning, no trend, and no record.
Existing alternatives are either full surveillance platforms (heavy, invasive)
or hardware counters (need extra sensors per lane).

**Problem:** operators need a fast, trustworthy answer to “how long is the
queue, is it getting worse, and is my camera working?” without installing
hardware beyond a standard camera.

**Hard product boundaries (deliberate):** no facial recognition, no passenger
identification, no physical sensors (ultrasonic/ToF/LiDAR), no MQTT/IoT, no
cloud processing. See §7.

## 3. Product Goals

| ID | Goal |
|----|------|
| **G-01** | An operator can read the queue situation (size, trend, wait, state) in a single glance. |
| **G-02** | Queue geometry (area + counting lines) is configurable at runtime through the UI — no code edits. |
| **G-03** | Counting metrics (entries, exits, occupancy, wait) stay correct under jitter, occlusion, and direction reversals. |
| **G-04** | The system runs unattended on commodity hardware with visible health signals (camera, FPS, data freshness). |
| **G-05** | Changes can be verified objectively: automated tests + an offline evaluator against ground truth. |

## 4. Success Metrics

No production baseline exists yet; targets marked **TBD** need data this
repository does not contain. Where the project has pre-defined its own gates
(evaluator thresholds), they are quoted as *defined targets*, not achievements.

| ID | Metric | Measurement | Target | Status |
|----|--------|-------------|--------|--------|
| SM-01 | Counting accuracy vs ground truth | `scripts/evaluate_counts.py` on ≥30 min representative footage, ≥50 labelled intervals: MAE, ±1/±2 accuracy, bias, coverage | Defined gates: MAE < 1.0, ±1 > 85%, ±2 > 95%, \|bias\| < 0.5, coverage ≥ 80% | **TBD — never measured on real airport footage** |
| SM-02 | Time for an operator to answer the core situational questions | Usable-test with operators | Design goal: 3–5 s (from the dashboard design brief); not user-tested | **TBD** |
| SM-03 | Pipeline real-time headroom | Sustained FPS and inference ms at the deployed resolution | ≥ 15 FPS sustained; inference ≈ 27 ms/frame on MPS was reported in earlier runs (indicative, not continuously measured) | **TBD baseline** |
| SM-04 | Regression safety | Test suites pass on every change | 100% of `make test` (308 unittest) + `pytest` (358) + `pnpm test` (40) green | **Implemented (measured 2026-10-23)** |

## 5. Target Users & Needs

| User | Needs | Where served |
|------|-------|--------------|
| **Security operator** (primary) | Glanceable state (EMPTY→CONGESTED), occupancy %, wait time, trend, camera health, event feed | Dashboard: QueueStatusPanel, MetricGrid, charts, WaitingTimePanel, ActivityLog, OperatorHealth chip |
| **Ops supervisor** | Historical view, throughput numbers, recordable evidence | History endpoint (1 Hz × 60 min), `--record` JSONL, `--record-samples` |
| **Integrator/developer** | Configure geometry, thresholds; add metrics/endpoints; verify changes | Mapping editor, `camera.yaml`, API, tests, evaluator |
| **(excluded) Passengers** | — | No passenger-facing feature exists |

## 6. User Stories

- **US-01:** As an operator, I want to see at a glance whether the queue is
  normal/busy/congested, so I can call extra staff early.
- **US-02:** As an operator, I want to see how many people are waiting and the
  average wait, so I can set expectations with passengers.
- **US-03:** As an integrator, I want to draw the queue area and IN/OUT counting
  lines on the live camera feed, so I can calibrate a lane without editing code.
- **US-04:** As an integrator, I want a wrong calibration to be rejected without
  breaking the running system, so the lane never goes dark.
- **US-05:** As a supervisor, I want entry/exit events recorded with timestamps,
  so I can analyse throughput offline.
- **US-06:** As a developer, I want a one-command start and automated tests, so
  I can trust changes I make to the analytics.
- **US-07:** As an operator, I want to know when the camera stops delivering
  frames, so I never act on stale numbers.

## 7. Scope

### In scope
Camera/video/RTSP input, person detection & tracking, queue ROI + directional
IN/OUT line analytics, rule-based state classification, event logging, local
FastAPI service, React operator dashboard (presentation only; the backend
is the single source of truth), interactive
mapping calibration with hot reload, offline evaluation tooling, dataset/ML
tooling for a future custom model.

### Out of scope (explicitly not built)
| Item | Reason |
|------|--------|
| Physical sensors (infrared, ToF, LiDAR, HC-SR04) | Camera-only scope |
| Multi-sensor fusion / motion-sensor engines, MQTT, IoT gateway, cloud | Out of scope — camera-only; the legacy `core/` sensor engine was removed 2026-10-22 (git history) |
| Facial recognition / passenger identification | Prohibited by product boundary |
| Production/deployment-grade accuracy claims | Prototype; accuracy gates not yet passed on real footage (SM-01) |
| People/m² physical density | Requires physical calibration |
| User accounts / authentication / database of people | No user system exists; all data is local and non-identifying |
| Multi-lane / multi-camera UI | API & mapping schema are camera-keyed, but the UI drives one camera (CAMERA 01) |

## 8. Functional Requirements

Format: **FR-ID — requirement — Status**. Related AC in §10.

### Detection & tracking
- **FR-01** — Ingest webcam index, video file, image, or RTSP/HTTP URL as the
  frame source, paced to a configurable FPS. — **Implemented**
  (FrameStream in `app/detect_people.py`; `--source`, `--fps`).
- **FR-02** — Detect people with YOLOv5s (COCO class 0), configurable
  confidence and input size. — **Implemented** (`--conf`, `--imgsz`, `--device`).
- **FR-03** — Track detections with SORT, yielding stable IDs across frames
  through occlusion. — **Implemented** (`app/sort_tracker.py`,
  `max_age/min_hits/iou_threshold`).

### Geometry & mapping
- **FR-04** — Configure the queue ROI (polygon) and separate IN/OUT counting
  lines at runtime, from the dashboard, over the raw camera feed. —
  **Implemented** (`MappingEditor.tsx`, `GET/PUT /api/dashboard/mapping`,
  tests `test_mapping.py`, `test_mapping_api.py`).
- **FR-05** — Store mapping geometry per camera in normalised coordinates
  (resolution-independent) with schema validation, atomic writes, and a `.bak`
  of the previous store. — **Implemented** (`app/mapping.py`,
  `config/mappings.yaml`).
- **FR-06** — Exactly one active geometry source: a valid mapping wins;
  legacy `camera.yaml` geometry is fallback only; never a combination. —
  **Implemented** (`mapping_source` field, resolver in `queue_prototype.py`).
- **FR-07** — A failed mapping update is rejected before any write, leaving the
  last known-good mapping running. — **Implemented** (PUT validates first;
  test `test_invalid_mapping_rejected_and_last_known_good_survives`).
- **FR-08** — Hot-reload a validated mapping into the running detector without
  restart, resetting per-track crossing state. — **Implemented**
  (`app/reload_signal.py`, verified live: “Config reloaded via api”).
- **FR-09** — Legacy pixel-based geometry (`--define-roi`, `camera.yaml`) keeps
  working for deployments without a mapping. — **Implemented** (legacy
  conversion incl. X-ray-side orientation fix for vertical lines).
- **FR-10** — Reference point for “is this person in the line/ROI” is
  configurable (`bottom_center` default, `centroid` legacy). — **Implemented**.

### Queue analytics
- **FR-11** — Occupancy = live ROI membership (reference point inside polygon),
  expressed as count and % of `queue_capacity`; never derived from
  entries − exits. — **Implemented** (regression tests in
  `TestMappingDrivenCrossings`).
- **FR-12** — Forward IN crossings register an entry; forward OUT crossings
  finalise exit time and waiting duration; opposite-direction crossings emit an
  explicit `line_crossing_reversed` event and change no counter. —
  **Implemented** (crossing engine in `queue_prototype.py`).
- **FR-13** — Commit a crossing only after N consecutive far-side frames
  (`crossing_debounce_frames`, default 2) with per-track/per-line dedup, so
  jitter never double-counts. — **Implemented** (separate knob from
  `classifier.debounce_frames`).
- **FR-14** — Entry/exit flow rates over configurable rolling windows
  (default 1 + 5 min). — **Implemented** (`flow_windows_minutes`).
- **FR-15** — Per-person waiting time from entry to exit; averages per snapshot. —
  **Implemented** (only OUT crossings finalise; test-covered).

### Classification & states
- **FR-16** — Classify queue state from occupancy ladder with hysteresis and
  debounce: EMPTY → SPARSE → NORMAL → CROWDED → CONGESTED, each snapshot
  carrying `reasons[]`. — **Implemented** (`RuleClassifier`).
- **FR-17** — CONGESTED additionally requires mean spacing ≤ configured px. —
  **Implemented** (`congested_max_spacing_px`, `congested_require_spacing`).
- **FR-18** — BACKLOG state when count/occupancy/flow/wait conditions are
  sustained and a flow baseline is configured. — **Partial**: code exists, but
  `backlog.flow.baseline_people_per_minute: null` disables it by default and no
  baseline is measured.
- **FR-19** — MOVING / STAGNANT / IDLE movement state from crossing count in a
  window. — **Implemented** (`movement:` config).
- **FR-20** — Congestion stopwatch with start/end events and duration in the
  snapshot. — **Implemented** (`CongestionStopwatch`).

### API & dashboard
- **FR-21** — FastAPI exposes snapshot, health, config, events, history, and
  MJPEG endpoints; the dashboard polls snapshot at 1 Hz. — **Implemented**
  (12 routes; see `docs/API.md`).
- **FR-22** — Raw MJPEG variant (`?annotated=0`) for calibration while the
  default annotated stream keeps existing consumers working. — **Implemented**.
- **FR-23** — History: 1 Hz samples kept 60 minutes (3600), returned evenly
  downsampled (`?points=2..3600`, default 240). — **Implemented**
  (`snapshot_buffer.py`, `GET /api/dashboard/history`).
- **FR-24** — Dashboard renders backend telemetry only: no synthetic data
  source, no scenario switch. Connectivity (backend unreachable / camera
  offline / no frames) is presented separately from the queue condition,
  which is always the backend classifier's answer. — **Implemented**
  (`telemetry.ts`, `App.tsx`).
- **FR-25** — Operator-facing event wording is plain language; technical
  wording reserved for diagnostics (`label`/`detail` vs `technical`). —
  **Implemented** (snapshot kind translation + `test_event_language.py`).
- **FR-26** — Per-state operational recommendations (`queue.action`) come from
  config, not invented by the frontend. — **Implemented**
  (`recommendations:` override table).
- **FR-27** — Graceful shutdown from the dashboard (`POST /api/shutdown`)
  stops loop and server cooperatively. — **Implemented** (`app/shutdown.py`).

### Measurement & ML tooling
- **FR-28** — Record events (`--record`) and 1 Hz occupancy samples
  (`--record-samples`) as JSONL. — **Implemented**.
- **FR-29** — Evaluate recorded samples against a ground-truth CSV with
  alignment rules and metrics (MAE, ±1/±2, bias, coverage) and exit codes. —
  **Implemented** (`scripts/evaluate_counts.py` + tests).
- **FR-30** — Dataset engineering workflow: contract, manifests, validation,
  leakage-safe split, SQLite catalog, experiment provenance, hard-example
  mining. — **Implemented** (`app/ml/*`, Makefile `dataset-*`/`catalog-*`
  targets, tests; see `docs/DATASETS.md`).
- **FR-31** — Train and evaluate a custom person-detection model. —
  **Partial**: tooling exists (`scripts/train_person.py`,
  `evaluate_model.py`, `collect_hard_examples.py`), but only the pretrained
  `yolov5s.pt` is in the repo — no custom-trained weights.
- **FR-32** — Physical calibration (metres) via homography or `px_per_metre`. —
  **Partial**: implemented in config path but default is `null` (PX ONLY);
  no homography file ships with the repo.
- **FR-33** — Face refinement of head detections (YuNet). — **Partial**:
  code + `models/face_detection_yunet_2023mar.onnx` present, default disabled
  (`--face` opt-in), not covered by the mapping-era test suites.

### Camera resilience & console presentation
- **FR-34** — A camera that fails to open or stops delivering frames must not
  stop the FastAPI service: the capture supervisor retries with bounded
  backoff (up to 5 attempts, waits 1/2/4/8 s between them), then stops and
  waits for a manual source selection (re-selecting a source while offline
  restarts the sequence). The snapshot reports `camera.status`
  (`OK`/`RECONNECTING`/`NO_FRAMES`/`OFFLINE`) and the real age of the last
  frame. — **Implemented** (`_open_live_stream` + loop supervisor in
  `detect_people.py`; `test_camera_controls.py`, `test_dashboard_adapter.py`).
- **FR-35** — Availability is never confused with classification: `scenario`
  summarises pipeline availability (`ONLINE`/`DEGRADED`/`OFFLINE`/`NO_DATA`),
  `queue.state` passes the classifier vocabulary through unchanged, and a
  measurement without a reading is `null` on the wire (never a fabricated
  zero; `0` appears only when a reading exists). — **Implemented**
  (`dashboard_adapter.py`, `telemetry.ts`; `telemetry.test.ts`).
- **FR-36** — Dashboard theme: dark by default, persistent light/dark toggle
  that survives reloads without a page reload, tokens applied everywhere. —
  **Implemented** (`:root.light` in `index.css`, toggle in `App.tsx`,
  no-flash script in `index.html`).

## 9. Non-Functional Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| NFR-01 | Local-only processing; no network egress beyond the dashboard↔API on localhost | **Implemented** (CORS limited to `:8443`; no external calls in `app/`) |
| NFR-02 | No identity data: no faces stored, no passenger identification | **Implemented** by design boundary |
| NFR-03 | Real-time: keep up with camera pace; expose measured FPS/timings | **Implemented** (per-frame timings + end-of-run summary) |
| NFR-04 | Crash-safe configuration: atomic writes, invalid input never replaces good config | **Implemented** (`save_mapping` + PUT ordering) |
| NFR-05 | Thread-safe snapshot handoff between pipeline and API | **Implemented** (`SnapshotHolder` lock) |
| NFR-06 | Deterministic, camera-free test suite for analytics | **Implemented** (308 unittest + 358 pytest, no hardware needed) |
| NFR-07 | Honest freshness: dashboard shows LIVE/STALE/OFFLINE from data age, not hope | **Implemented** (`freshness`, `last_frame_age_ms`; camera `NO SIGNAL` in launcher) |
| NFR-08 | Documentation follows code; claims require evidence | **Implemented as a process rule** (this PRD + `AGENTS.md`) |

## 10. Acceptance Criteria

- **AC-01 (FR-04, FR-07)** — Given the mapping editor is open,
  When the operator draws a degenerate ROI (zero area) and presses Save,
  Then the API responds `applied: false` with an error, `mappings.yaml` on disk
  is byte-identical to before, and the detector keeps the previous mapping.
  *Evidence: `test_mapping_api.py`, live API check 2026-10-21.*
- **AC-02 (FR-04, FR-08)** — Given a valid mapping is edited in the UI,
  When the operator presses Save,
  Then `mappings.yaml` is rewritten atomically with a `.bak`, the detector logs
  the reload on its next frame, and new crossings use the new geometry.
  *Evidence: live end-to-end run 2026-10-21.*
- **AC-03 (FR-12)** — Given a person walks out of the queue (reverse across
  IN/OUT), Then a `line_crossing_reversed` event appears and `total_entries` /
  `total_exits` do not change. *Evidence: `TestMappingDrivenCrossings`.*
- **AC-04 (FR-13)** — Given a track jitters across a line for 1 frame,
  Then no crossing event is emitted (default debounce 2); with
  `crossing_debounce_frames: 1` the legacy single-frame behaviour fires.
  *Evidence: `test_queue_prototype.py`.*
- **AC-05 (FR-11)** — Given people enter via the IN line,
  Then occupancy follows live ROI membership and `exit_unix` is untouched until
  a forward OUT crossing. *Evidence: same suite.*
- **AC-06 (FR-21..24)** — Given the backend runs with `--api`,
  When the dashboard is opened,
  Then snapshot polls at 1 Hz, MJPEG plays (annotated and `?annotated=0`),
  and the browser console has no errors (favicon 404 excepted).
  *Evidence: manual verification procedure in `docs/DEVELOPMENT.md`.*
- **AC-07 (FR-29)** — Given samples recorded from a video and a ground-truth
  CSV, When the evaluator runs, Then it reports MAE/±1/±2/bias/coverage and
  exits 0/1/2 per data sufficiency. *Evidence: `test_evaluate_counts.py`.*
- **AC-08 (G-05)** — Given any code change, When `make test`, `pytest`, and
  `pnpm test` run, Then all pass. *Evidence: 2026-10-23 runs — `make test`:
  308 (unittest) + 358 (pytest); `pnpm test`: 40.*

## 11. Feature Implementation Status (summary)

| Area | Status | Evidence |
|------|--------|----------|
| Detection / tracking pipeline | Implemented | runtime + tests |
| ROI + IN/OUT counting, debounce, reversed-crossing rules | Implemented | `TestMappingDrivenCrossings` (10) |
| Interactive mapping editor + API + hot reload | Implemented | `test_mapping.py` (25), `test_mapping_api.py` (7), live UI verification |
| Occupancy / flow / waiting / movement / congestion | Implemented | `test_queue_prototype.py` |
| Classifier EMPTY→CONGESTED | Implemented | config thresholds + tests |
| BACKLOG state | Partial (disabled: no flow baseline) | `baseline_people_per_minute: null` |
| Dashboard (API-only: charts, events, diagnostics, themes) | Implemented | build + browser verification |
| Camera failure resilience (bounded reconnect, API stays up) | Implemented | supervisor logic + `test_camera_controls.py` / `test_dashboard_adapter.py` |
| Honest availability states (nulls; connectivity vs condition) | Implemented | adapter tests + `telemetry.test.ts` |
| Mapping raw stream `?annotated=0`, annotated default | Implemented | live stream byte checks |
| History buffer (60 min @1 Hz) | Implemented | `snapshot_buffer.py` |
| Offline evaluator | Implemented | `test_evaluate_counts.py` |
| Dataset/ML tooling | Implemented | `ml/*` + Makefile targets + dataset tests |
| Custom trained model | Partial (tooling only) | only `yolov5s.pt` in `models/` |
| Metric calibration (metres) | Partial (off by default) | both calibration fields `null` |
| Face refinement | Partial (opt-in, untested in suites) | YuNet present, default off |
| Multi-camera UI | Planned (schema is camera-keyed; UI drives CAMERA 01) | `MappingEditor` reads one camera |
| Sensors / MQTT / cloud / facial recognition | Planned — explicitly out of scope | §7 |
| Accuracy gates on real footage | Unknown — never measured here | SM-01 TBD |

## 12. Constraints, Assumptions, Unknowns

**Constraints**
- Python 3.9, Node 18+ (dev runs on Node 22 / pnpm via corepack); Apple
  Silicon MPS or CPU. No GPU/cluster assumed.
- Single machine, single lane; localhost networking only.
- Product boundary list in §7 is fixed unless explicitly changed.

**Assumptions**
- Scene framing stays stable enough that normalised geometry remains valid
  across resolution changes.
- Operator supplies a camera whose framing matches what the mapping was drawn
  on.

**Unknowns (need your decision/data)**
1. **Accuracy on real airport footage** — no ground truth exists in this repo
   (SM-01). Needs labelled footage.
2. **Flow baseline for BACKLOG** — needs a measured
   `baseline_people_per_minute` (FR-18).
3. **Operator usability** — the 3–5 s glance goal was never user-tested (SM-02).
4. **Multi-camera roadmap** — schema/API are camera-keyed; product priority for
   a multi-lane UI is undecided.
5. **Custom model** — whether to invest in fine-tuning (tooling ready,
   FR-31) depends on the accuracy gap measured in (1).


