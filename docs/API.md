# API Reference

## Overview

The FastAPI server provides real-time queue monitoring data for the React dashboard and external integrations.

**Base URL:** `http://localhost:8080` (configurable via `--api-port`)

**CORS Origins:** `http://localhost:8443`, `http://127.0.0.1:8443`

**Content-Type:** `application/json` (except MJPEG stream)

---

## Endpoints

### GET /api/snapshot

**Purpose:** Latest queue snapshot for dashboard polling (1 Hz target)

**Response:** `QueueSnapshot` (see schema below)

**Update Behavior:** Updated each frame by camera engine (~10-30 Hz). Dashboard polls at 1 Hz.

**Errors:** Returns `null` if no data available yet (camera not started).

```json
{
  "timestamp": "2026-09-02T02:20:32.229Z",
  "frame_id": 7519,
  "freshness": "LIVE",
  "staleness_ms": 66,
  "fps": 11.11,
  "inference_ms": 19.65,
  "tracking_ms": 0.35,
  "analytics_ms": 0.25,
  "detection_count": 1,
  "tracked_people": 1,
  "device": "mps",
  "people_in_queue": 0,
  "total_in": 0,
  "total_out": 0,
  "queue_state": "UNCONFIGURED",
  "occupancy_percent": 0.0,
  "entry_rate": 0.0,
  "exit_rate": 0.0,
  "avg_wait_s": 0.0,
  "longest_wait_s": 0.0,
  "congestion_duration_s": 0.0,
  "spacing_px": null,
  "movement": "IDLE",
  "config_status": {
    "roi_configured": false,
    "line_configured": false,
    "calibrated": false,
    "classifier_enabled": false
  },
  "calibration": "PX ONLY",
  "camera_state": {
    "link": "ONLINE",
    "fps": 14.99,
    "resolution": "?x?",
    "frame_age_ms": 66
  },
  "tracks": [],
  "events": []
}
```

---

### GET /api/health

**Purpose:** Camera and CV pipeline health status

**Response:** `HealthResponse`

**Data Source:** Derived from latest snapshot

```json
{
  "camera": {
    "link": "ONLINE",
    "fps": 15.23,
    "resolution": "?x?",
    "frame_age_ms": 65
  },
  "cv": {
    "detection": "OK",
    "tracking": "OK",
    "analytics": "OK",
    "inference_ms": 21.45,
    "model": "YOLOv5s",
    "tracker": "SORT",
    "detections": 1,
    "active_tracks": 1
  },
  "recording": {
    "active": false
  },
  "data": {
    "status": "LIVE",
    "staleness_ms": 65
  }
}
```

**Status Values:**
- `camera.link`: `"ONLINE"` | `"OFFLINE"` | `"DEGRADED"`
- `cv.*`: `"OK"` | `"DEGRADED"` | `"STALLED"`
- `data.status`: `"LIVE"` | `"STALE"` | `"OFFLINE"` | `"NO_DATA"`

---

### GET /api/config

**Purpose:** Current camera configuration (read-only)

**Response:** JSON object with camera config

```json
{
  "camera_id": "CAMERA 01",
  "width": 0,
  "height": 0,
  "roi_configured": false,
  "line_configured": false,
  "xray_side": null,
  "roi_polygon": null,
  "counting_line": null,
  "classifier": {
    "occupancy_sparse": 0.1,
    "occupancy_normal": 0.4,
    "occupancy_crowded": 0.6,
    "occupancy_congested": 0.8,
    "hysteresis": 0.05,
    "debounce_frames": 5
  },
  "backlog": {
    "min_people": 15,
    "min_occupancy": 0.6,
    "min_wait_seconds": 300.0,
    "persistence_seconds": 10.0
  }
}
```

**Notes:**
- `width`/`height` = 0 means native resolution
- `xray_side`: `"left"` | `"right"` | `null` (derived from `xray_side_point`)
- `roi_polygon`: `[[x, y], ...]` or `null`
- `counting_line`: `[[x1, y1], [x2, y2]]` or `null`

---

### GET /api/events

**Purpose:** Recent events (for timeline/history)

**Query Parameters:**
- `limit` (optional, default 50, max 200)

**Response:** `{ "events": EventInfo[] }`

```json
{
  "events": [
    {
      "id": 42,
      "timestamp": 1725157232123,
      "kind": "exit",
      "severity": "info",
      "label": "Passenger cleared screening",
      "detail": "12 → 11"
    },
    {
      "id": 41,
      "timestamp": 1725157230500,
      "kind": "entry",
      "severity": "info",
      "label": "Passenger entered queue",
      "detail": "11 → 12"
    }
  ]
}
```

**Event Kinds:** `"entry"` | `"exit"` | `"state"` | `"congestion"` | `"system"`

**Severities:** `"info"` | `"warn"` | `"critical"` | `"good"`

---

### GET /api/camera/stream

**Purpose:** MJPEG stream of camera frames

**Response:** `multipart/x-mixed-replace; boundary=frame`

**Query Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `annotated` | `1` | `1` streams the operator view with detections, ROI and counting line burned in. `0` streams the same raw camera picture without annotations (used by the mapping editor, which renders its own geometry). |

**Usage:**
```html
<img src="http://localhost:8080/api/camera/stream" alt="Live camera" />
<!-- calibration editor: raw pixels, no burned-in geometry -->
<img src="http://localhost:8080/api/camera/stream?annotated=0" alt="Calibration" />
```

**Stream Format:**
```
--frame
Content-Type: image/jpeg
Content-Length: 245586

<JPEG binary data>
--frame
Content-Type: image/jpeg
Content-Length: 238912

<JPEG binary data>
...
```

**Frame Rate:** ~30 FPS (limited by camera + encoding)

**Error Handling:** If no frame available, stream waits (async sleep 1/30s).

---

## Dashboard endpoints

These are the endpoints the React dashboard actually consumes. They are
additive to the `/api/*` endpoints above and return the
`DashboardSnapshot` shape.

### GET /api/dashboard/snapshot

**Purpose:** The dashboard's single source of truth (polled at 1 Hz).
**Response:** `DashboardSnapshot` — see the model table below.

#### DashboardSnapshot

| Field | Type | Description |
|-------|------|-------------|
| `scenario` | string | `"NORMAL"` \| `"BUSY"` \| `"CONGESTED"` \| `"BACKLOG"` \| `"DEGRADED"` \| `"OFFLINE"` |
| `queue` | QueueTelemetry | Queue occupancy, flow, waiting and trend |
| `system` | SystemTelemetry | Pipeline performance |
| `camera` | CameraTelemetry | Camera link and health |
| `recording` | RecordingTelemetry | Recording subsystem state |
| `last_known_queue` | integer | Last trustworthy queue reading (used while offline) |
| `last_update` | integer | Epoch ms of the snapshot |
| `history` | Sample[] | Rolling history (240 samples) |
| `events` | EventInfo[] | Recent operator events |
| `tracks` | TrackInfo[] | Normalized track boxes (0..1) |
| `frame_ts` | integer | Epoch ms of the frame the snapshot was built from |

#### SystemTelemetry

| Field | Type | Description |
|-------|------|-------------|
| `inference_ms` | number | Measured inference time per frame |
| `model` / `tracker` | string | `"YOLOv5s"` / `"SORT"` |
| `detections` / `active_tracks` | integer | Detection and track counts |
| `detection_status` / `tracking_status` / `analytics_status` | string | `"OK"` \| `"DEGRADED"` \| `"STALLED"` |
| `device` | string | Compute device the inference ran on (`"mps"`, `"cpu"`, …) |

#### CameraTelemetry

| Field | Type | Description |
|-------|------|-------------|
| `link` | string | `"ONLINE"` \| `"OFFLINE"` \| `"DEGRADED"` |
| `fps` | number | Measured capture rate |
| `resolution` | string | `"WxH"` |
| `latency_ms` | number \| **null** | `null` when the pipeline does not measure it (this build has no network hop to time). A `0` would be indistinguishable from a real measurement. |
| `packet_loss` | number \| **null** | `null` when not measured |
| `last_frame_age_ms` | integer | **Measured** age of the newest captured frame |
| `uptime_sec` | integer | Seconds since the server started |

#### RecordingTelemetry

| Field | Type | Description |
|-------|------|-------------|
| `active` | boolean | Whether a recording is running |
| `since_sec` | number \| **null** | `null` when there is no recording subsystem |
| `size_gb` | number \| **null** | `null` when not measured |

### GET /api/dashboard/config

**Purpose:** Read-only camera configuration (ROI, counting line, calibration).
**Response:** `camera_id`, `roi_configured`, `line_configured`, `roi_polygon`,
`counting_line`, `xray_side`, `width`, `height`, `calibrated` (boolean) and
`calibration` (`"METRES"` | `"PX ONLY"`).

**Additive mapping fields (interactive calibration):** `source_shape_w`,
`source_shape_h` (frame the geometry is calibrated against), `mapping_active`
(true when the running analytics is driven by a validated mapping) and
`reference_point` (`"bottom_center"` | `"centroid"`). Clients that do not know
about them simply ignore them; nothing in the pre-mapping response changed.

### GET /api/dashboard/mapping

**Purpose:** The active per-camera mapping in normalised frame coordinates
(`x`, `y` ∈ `[0, 1]`), ready for the mapping editor.

**Response:** `camera_id`, `mapping_active`, `reference_point`,
`mapping_source` (`"mappings.yaml"` | `"legacy"` | `"none"`), `source_shape_w/h`
and `mapping` — the `MappingGeometry` document (`roi` polygon, `lines`
(`in` / `out`, each two points and a direction), `reference_point`,
`created_at_iso` / `updated_at_iso`). `mapping` is `null` when the camera is
not calibrated at all. When no valid file entry exists but `camera.yaml` has
legacy geometry, the in-memory derivation is returned with
`mapping_source: "legacy"`. Note the never-combined rule: a valid mapping is
the single active geometry source; legacy `roi` / `counting_line` /
`xray_side_point` are only used as an in-memory fallback, never merged.

### PUT /api/dashboard/mapping

**Purpose:** Validate, apply and persist a per-camera mapping (interactive
calibration). Camera-capable by design; the UI currently drives
`CAMERA 01` only.

**Request body:** `schema_version`, `roi` (polygon, normalised), `lines`
(both shapes accepted: `{"in": {...}, "out": {...}}` keys or
`[{id, points, direction}, ...]`), optional `reference_point` and
`source_shape` (drawn-against frame, for pixel-scale checks).

**Ordering guarantees (see `docs/ARCHITECTURE.md` § mapping):**

1. Validation happens before anything is written. An invalid payload never
   touches disk, so the running pipeline keeps its last known-good mapping —
   the response then is `{"applied": false, "errors": [...]}` (HTTP 200).
2. Persistence is atomic (`app/mapping.py.save_mapping`): serialise →
   re-parse the rendered document → tempfile → `os.replace`, plus a
   `mappings.yaml.bak` of the previous content.
3. Only a successful write asks the detection loop to hot-reload
   (`app/reload_signal.py`); the loop re-reads the config, swaps the mapping
   and resets per-track crossing state. A failed update therefore leaves the
   last known-good mapping active.

**Response:** `applied`, `camera_id`, `errors`, `backup_path` (only when an
existing store was replaced), `mapping_active`, `reference_point`.

### GET /api/dashboard/health

**Purpose:** Liveness of the API and the camera.
**Response:** `status`, `camera`, `api`, `last_frame_age_ms`.

### GET /api/dashboard/history

**Purpose:** The rolling history buffer as plain samples.
**Response:** `{"samples": [Sample, ...]}` — at most 240 entries.

### POST /api/shutdown

**Purpose:** Stop the backend from the dashboard.
**Response:** `{"status": "shutting_down"}`

**Behaviour:** the request asks the process to shut down *cooperatively*.
The detection loop stops at the next frame, releases the camera, and the
API server stops accepting connections; in-flight requests are allowed to
finish. A watchdog terminates the process if the graceful path has not
completed within 10 s, and says so in the log. The endpoint never calls
`os._exit()` directly, so no cleanup step is skipped.

---

## Data Models

### QueueSnapshot

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO8601 UTC with milliseconds, trailing Z |
| `frame_id` | integer | Monotonic frame counter |
| `freshness` | string | `"LIVE"` \| `"STALE"` \| `"OFFLINE"` |
| `staleness_ms` | integer | Milliseconds since last frame |
| `fps` | number | Measured pipeline FPS |
| `inference_ms` | number | YOLO inference time per frame |
| `tracking_ms` | number | SORT tracking time per frame |
| `analytics_ms` | number | Queue analytics time per frame |
| `detection_count` | integer | Raw YOLO detections this frame |
| `tracked_people` | integer | Active SORT tracks |
| `people_in_queue` | integer | Tracks with centroid in ROI |
| `total_in` | integer | Cumulative ENTER crossings |
| `total_out` | integer | Cumulative EXIT crossings |
| `queue_state` | string | `"NORMAL"` \| `"BUSY"` \| `"CONGESTED"` \| `"BACKLOG"` \| `"DATA_UNAVAILABLE"` |
| `occupancy_percent` | number | people_in_queue / queue_capacity × 100 |
| `entry_rate` | number | ENTER crossings per minute (1-min window) |
| `exit_rate` | number | EXIT crossings per minute (1-min window) |
| `avg_wait_s` | number | Mean entry→exit duration (seconds) |
| `longest_wait_s` | number | Max entry→exit duration (seconds) |
| `congestion_duration_s` | number | Seconds in CONGESTED/BACKLOG state |
| `spacing_px` | number \| null | Mean pairwise centroid distance (pixels) |
| `movement` | string | `"MOVING"` \| `"STAGNANT"` \| `"IDLE"` |
| `config_status` | ConfigStatus | ROI/line/calibration status |
| `calibration` | string | `"METRES"` \| `"PX ONLY"` |
| `device` | string | Compute device the inference ran on (`"mps"`, `"cpu"`, …); empty when unknown |
| `staleness_ms` | int | **Measured** age of the newest captured frame; `999999` when there is no signal |
| `camera_state` | CameraState | Camera link status |
| `tracks` | Track[] | Normalized track boxes (0..1) |
| `events` | Event[] | Recent events |

### Track

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | SORT track ID (stable across frames) |
| `x` | number | Center X / frame width (0..1) |
| `y` | number | Center Y / frame height (0..1) |
| `w` | number | Box width / frame width (0..1) |
| `h` | number | Box height / frame height (0..1) |
| `in_roi` | boolean | Centroid inside queue ROI |

### Event

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Event sequence ID |
| `t` | integer | Epoch milliseconds |
| `kind` | string | `"entry"` \| `"exit"` \| `"state"` \| `"congestion"` \| `"system"` |
| `severity` | string | `"info"` \| `"warn"` \| `"critical"` \| `"good"` |
| `label` | string | Human-readable label |
| `detail` | string \| null | Additional detail (e.g., "12 → 11") |

### ConfigStatus

| Field | Type | Description |
|-------|------|-------------|
| `roi_configured` | boolean | ROI polygon has ≥3 points, area > 0 |
| `line_configured` | boolean | Counting line has 2 endpoints |
| `calibrated` | boolean | Homography or px_per_metre configured |
| `classifier_enabled` | boolean | ROI configured (enables classifier) |

### CameraState

| Field | Type | Description |
|-------|------|-------------|
| `link` | string | `"ONLINE"` \| `"OFFLINE"` \| `"DEGRADED"` |
| `fps` | number | Camera capture FPS |
| `resolution` | string | `"WxH"` format |
| `frame_age_ms` | integer | Estimated frame interval (ms) |

---

## Frontend Integration

### Vite Proxy (Development)

```typescript
// vite.config.ts
proxy: {
  '/api': {
    target: 'http://localhost:8080',
    changeOrigin: true,
    rewrite: (path) => path,  // Keep /api prefix
  },
}
```

Dashboard fetches `/api/snapshot` → Vite proxies to `http://localhost:8080/api/snapshot`

### TypeScript Client (src/lib/api.ts)

```typescript
// Polling at 1 Hz
const apiSnap = await getSnapshot();  // GET /api/snapshot
const config = await getConfig();     // GET /api/config
const health = await getHealth();     // GET /api/health

// MJPEG stream
<img src={getVideoStreamUrl()} />     // /api/camera/stream
```

### React Hook (src/lib/telemetry.ts)

```typescript
const { snap, error, refetch } = useRealTelemetry(1000); // 1 Hz poll
// Returns internal Snapshot type (converted from API response)
```

---

## Error Handling

| Scenario | HTTP Status | Response |
|----------|-------------|----------|
| No snapshot yet | 200 | `null` (not an error) |
| Internal conversion error | 500 | `{"detail": "Internal Server Error"}` |
| Invalid query param | 422 | `{"detail": [...]}` (FastAPI validation) |
| CORS blocked | — | Browser blocks (check origins) |

---

## Testing Endpoints

```bash
# Health check
curl -s http://localhost:8080/api/health | python -m json.tool

# Snapshot
curl -s http://localhost:8080/api/snapshot | python -m json.tool

# Config
curl -s http://localhost:8080/api/config | python -m json.tool

# Events
curl -s http://localhost:8080/api/events | python -m json.tool

# MJPEG (5 sec test)
curl -N --max-time 5 http://localhost:8080/api/camera/stream | head -10
```

---

## Versioning

- API version: `1.0.0` (in FastAPI app metadata)
- No formal versioning in URL yet — breaking changes require dashboard update
- Pydantic models define contract; changes require frontend rebuild