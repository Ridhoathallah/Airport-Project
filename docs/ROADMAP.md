# Roadmap

## Purpose

This document explicitly separates **CURRENT IMPLEMENTATION** from **FUTURE ROADMAP** to prevent premature implementation of planned features.

> **Rule:** Do not implement, document, or claim any "Future" item as current functionality without explicit approval.

---

## Current Implementation (Camera-Only Prototype)

**Status:** ✅ Running, tested, documented

| Component | Status | Evidence |
|-----------|--------|----------|
| Webcam / video / image / RTSP input | ✅ | `detect_people.py --source` |
| YOLOv5s person detection (COCO class 0) | ✅ | `models/yolov5s.pt` loaded |
| SORT tracking (Kalman + IoU) | ✅ | `app/sort_tracker.py` |
| Queue ROI polygon | ✅ | `config/camera.yaml roi` |
| Interactive ROI definition | ✅ | `--define-roi` |
| Interactive mapping editor | ✅ | Dashboard "Edit mapping" (`MappingEditor.tsx`), `GET/PUT /api/dashboard/mapping` |
| `config/mappings.yaml` store | ✅ | Per-camera normalised geometry, atomic writes + `.bak` |
| Counting line + direction | ✅ | `counting_line`, `xray_side_point` (legacy fallback while a mapping is active) |
| ENTER/EXIT events | ✅ | Line crossing logic |
| Occupancy % (people/capacity) | ✅ | `QueueAnalytics` |
| Flow rates (1-min, 5-min) | ✅ | `flow_windows_minutes` |
| Waiting time (entry→exit) | ✅ | Per-track timer |
| Pixel spacing (centroid distances) | ✅ | Pairwise Euclidean |
| Rule-based classifier | ✅ | `RuleClassifier` (EMPTY→BACKLOG) |
| Hysteresis + debounce | ✅ | Configurable thresholds |
| Congestion stopwatch | ✅ | `CongestionStopwatch` |
| Event logging (JSONL) | ✅ | `--record` |
| Monitoring dashboard (OpenCV) | ✅ | `draw_dashboard()` |
| Legacy HUD | ✅ | `--legacy-hud` |
| FastAPI server | ✅ | `app/api_server.py` |
| MJPEG stream | ✅ | `/api/camera/stream` |
| Snapshot architecture | ✅ | `snapshot.py`, `snapshot_holder.py` |
| React/Vite dashboard | ✅ | `Queue-monitoring-dashboard/` |
| Simulator mode | ✅ | `useTelemetry()` |
| Live API mode | ✅ | `useRealTelemetry()` |
| CameraPanel MJPEG + overlays | ✅ | `sections.tsx` |
| Vite proxy `/api` → `:8080` | ✅ | `vite.config.ts` |
| Evaluation tooling | ✅ | `scripts/evaluate_counts.py` |
| Unit tests (297 `make test` / 350 pytest) | ✅ | `make test` |

---

## Current Phase: Camera-Only Prototype Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT VALIDATION PHASE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Real Airport CCTV Footage
         │
         ▼
Frame Capture & Processing
         │
         ▼
Manual Ground Truth Annotation
         │
         ▼
Evaluation (MAE, ±1, ±2 accuracy, Bias)
         │
         ▼
Threshold Calibration
         │
         ▼
Dataset Creation (if needed)
         │
         ▼
Fine-tuning (if pretrained model fails gate)
```

**Goal:** Validate camera-only prototype against representative airport CCTV footage before any multi-sensor work.

---

## Future Roadmap (NOT IMPLEMENTED)

### Phase 2: Multi-Sensor Validation (Future)

```
Camera Analytics (this prototype)
         │
         ▼
Sensor 1 (HC-SR04 / infrared / ToF at entry)
         │
         ▼
Sensor 2 (validation at processing line)
         │
         ▼
Sensor 3 (downstream validation)
         │
         ▼
Sensor Fusion
         │
         ▼
MQTT / IoT Gateway
```

### Explicitly NOT Implemented (Do Not Build)

| Feature | Status | Notes |
|---------|--------|-------|
| HC-SR04 ultrasonic sensor | ❌ NOT IMPLEMENTED | Requires hardware |
| Infrared beam sensors | ❌ NOT IMPLEMENTED | Requires hardware |
| Time-of-Flight (ToF) sensors | ❌ NOT IMPLEMENTED | Requires hardware |
| LiDAR | ❌ NOT IMPLEMENTED | Requires hardware |
| Physical Sensor 1 / 2 / 3 | ❌ NOT IMPLEMENTED | Future phase |
| Multi-sensor fusion | ❌ NOT IMPLEMENTED | `core/fusion.py` exists but dormant |
| MQTT broker / client | ❌ NOT IMPLEMENTED | Future IoT phase |
| IoT gateway / edge device | ❌ NOT IMPLEMENTED | Future |
| Cloud processing / API | ❌ NOT IMPLEMENTED | "Everything runs locally" |
| Airport production deployment | ❌ NOT IMPLEMENTED | Prototype only |
| Facial recognition | ❌ NOT IMPLEMENTED | Explicitly prohibited |
| Passenger identification | ❌ NOT IMPLEMENTED | Track IDs are ephemeral |
| Physical people/m² density | ❌ NOT IMPLEMENTED | Requires calibration |
| Backlog classification | ⚠️ DISABLED | `baseline_people_per_minute: null` |
| Fine-tuned YOLO model | ❌ NOT DONE | Start with pretrained YOLOv5s |

### Legacy Core Engine (Dormant)

The `core/` directory contains a complete multi-sensor engine from a previous phase:

| Module | Purpose | Status |
|--------|---------|--------|
| `core/monitor.py` | QueueMonitor (Sensor 1/2/3 ingestion) | Dormant |
| `core/fusion.py` | Sensor fusion logic | Dormant |
| `core/backlog.py` | Backlog detection | Dormant |
| `core/alerts.py` | Alert engine | Dormant |
| `core/state_machine.py` | Zone state machine | Dormant |
| `core/simulation.py` | Sensor simulation | Dormant |
| `core/models.py` | SensorReading, SensorHealth | Dormant |
| `core/config.py` | Legacy YAML config | Dormant |
| `core/metrics.py` | Legacy metrics | Dormant |

**Activation:** Only with `--monitor` flag (opt-in).
**Current role:** Reference for future sensor-fusion phase. Do not modify.

---

## Development Guardrails

The following rules prevent scope creep and hallucinated requirements:

### Must Never Change Without Explicit Approval

1. **Project Purpose** — Camera-only queue analytics for airport X-ray screening
2. **Vision Layer** — YOLOv5s + SORT (do not replace with other detectors/trackers)
3. **Backend = Source of Truth** — Frontend must not duplicate queue business logic
4. **No Facial Recognition** — System detects/tracks people, does not identify them
5. **Local-Only Processing** — No cloud uploads, no external services
6. **Camera-Only Scope** — No sensor fusion unless explicitly instructed

### Must Not Do

- ❌ Implement HC-SR04, infrared, ToF, LiDAR, or any physical sensors
- ❌ Add MQTT, IoT gateway, or cloud integration
- ❌ Claim "deployment-grade" or "production-ready" accuracy
- ❌ Treat simulator data as production data
- ❌ Fabricate telemetry in frontend
- ❌ Remove `--monitor` / `core/` without verifying no dependencies
- ❌ Replace YOLO/SORT without approval
- ❌ Duplicate classifier/flow/waiting logic in React components
- ❌ Change classifier thresholds without calibration data

### Must Do After Changes

- ✅ Run `make test` (297 tests)
- ✅ Run `make smoke` (offline validation)
- ✅ Validate with real camera when camera behavior affected
- ✅ Update documentation in same commit
- ✅ Keep `README.md` "Last reviewed against code" date current

---

## Dataset / Training Policy

| Step | Policy |
|------|--------|
| 1. Start | Use pretrained YOLOv5s (`models/yolov5s.pt`) |
| 2. Validate | Test on representative airport CCTV footage |
| 3. Gate | If MAE/±1/±2 meets threshold → keep pretrained |
| 4. Fail gate | Collect frames, annotate, fine-tune, re-evaluate |
| 5. Tools | Roboflow / AnyLabeling **may** be used for annotation (not runtime deps) |

> **No airport-specific training has been done.** All detection uses COCO-pretrained weights.

---

## Accuracy / Claims Policy

> The project must **never** claim detection accuracy based only on visual inspection.

**Accuracy claims require:**
1. Representative labeled footage
2. Measured evaluation results from `scripts/evaluate_counts.py`

**Use language:**
- ✅ "Observed / verified in testing"
- ✅ "Measured MAE of X on Y footage"
- ❌ "Deployment-grade"
- ❌ "Production ready"
- ❌ "95% accurate" (without evaluation results)

---

## Next Steps (Priority Order)

1. **Obtain representative airport CCTV footage** — blocker for validation
2. **Run evaluation pipeline** — `scripts/evaluate_counts.py` with ground truth
3. **Calibrate classifier thresholds** — based on evaluation results
4. **Create dataset if needed** — only if pretrained model fails gate
5. **Fine-tune YOLO if needed** — only after dataset creation

> **Do not proceed to Phase 2 (multi-sensor) until camera-only prototype is validated on real airport footage.**