# Integration Changelog

## Purpose

This document records all bug fixes and improvements made during the end-to-end integration of the YOLO computer vision pipeline with the React dashboard. Each entry describes the problem, root cause, fix, and verification.

---

## Fix 1: API Endpoint Path Mismatch

**Date:** Integration session  
**File:** `Queue-monitoring-dashboard/src/lib/api.ts:119`  
**Severity:** Critical — dashboard received stale OFFLINE data

### Problem

The React dashboard fetched snapshot data from `/api/snapshot`, but the FastAPI server's dashboard-specific endpoint was registered at `/api/dashboard/snapshot`. The `/api/snapshot` endpoint existed but returned an older response format that did not include the full dashboard telemetry structure. As a result, the dashboard displayed stale `OFFLINE` status with placeholder values even when the backend was running correctly.

### Root Cause

The `getSnapshot()` function in `api.ts` was hardcoded to call `/snapshot` (line 119), which resolved through the Vite proxy to `http://localhost:8080/api/snapshot`. The correct endpoint for the dashboard-shaped response was `/dashboard/snapshot`.

### Fix

```typescript
// Before
return fetchJson<SnapshotResponse>("/snapshot");

// After
return fetchJson<SnapshotResponse>("/dashboard/snapshot");
```

### Verification

After the fix, `curl -s http://localhost:8080/api/dashboard/snapshot` returned a full `DashboardSnapshot` with `camera.link: "ONLINE"`, live queue data, history samples, and events. The dashboard displayed real-time status instead of stale OFFLINE data.

---

## Fix 2: White Screen Crash on Empty Chart Data

**Date:** Integration session  
**File:** `Queue-monitoring-dashboard/src/components/charts.tsx` (lines 50, 71, 108)  
**Severity:** Critical — dashboard rendered blank white screen

### Problem

When the dashboard first loaded, the history buffer was empty. The `Sparkline`, `OccupancyChart`, and `FlowChart` components received empty arrays and crashed during rendering, causing a white screen with no visible error in the browser console.

### Root Cause

Each chart component attempted to compute `Math.max(...data)` or iterate over `data` without checking whether the array was empty. An empty spread (`Math.max()` with no arguments) returns `-Infinity`, and subsequent arithmetic produced `NaN` values that broke SVG rendering.

### Fix

Added early-return guards at the top of each component:

```tsx
// Sparkline (line 50)
if (!data.length) return <svg viewBox={`0 0 ${w} ${h}`} ... />;

// OccupancyChart (line 71)
if (!data.length) return <div className="relative h-full w-full" />;

// FlowChart (line 108)
if (!data.length) return <div className="relative h-full w-full" />;
```

### Verification

Dashboard loaded without crashing. Charts remained empty until the first history sample arrived from the backend, then populated progressively.

---

## Fix 3: History Buffer Never Populated

**Date:** Integration session  
**File:** `airport_people_counter/app/detect_people.py` (after line 1043)  
**Severity:** High — dashboard trend charts showed no historical data

### Problem

The dashboard received live snapshots but the `history` array was always empty. The `OccupancyChart` and `FlowChart` displayed only the current point with no trend line.

### Root Cause

The `snapshot_holder` was updated each frame with the latest snapshot, but the `HistoryBuffer` (a separate ring buffer used by the `/api/dashboard/history` endpoint) was never written to. The pipeline updated the holder but had no code to append samples to the history buffer.

### Fix

Added a `HistoryBuffer.append()` call immediately after the snapshot holder update:

```python
# detect_people.py, after line 1043
snap, _ = SnapshotBuilder.build(dash, system, camera_config, tracks, recent_events, stream_frame)
get_holder().update(snap, stream_frame)

# Append to history buffer for dashboard trend charts
from snapshot_buffer import get_history_buffer
from dashboard_models import Sample
now_ms = int(time.time() * 1000)
get_history_buffer().append(Sample(
    t=now_ms,
    queue=dash.queue.people_in_queue,
    occupancy=dash.queue.occupancy_percent or 0.0,
    entry_rate=dash.queue.entry_flow,
    exit_rate=dash.queue.exit_flow,
    avg_wait=round(dash.queue.avg_wait_s / 60.0, 1),
))
```

### Verification

After the fix, `/api/dashboard/history` returned up to 240 samples (4 minutes of data at 1 Hz). The dashboard charts displayed trend lines that updated in real time.

---

## Fix 4: `recent_events` NameError on First Event

**Date:** Integration session  
**File:** `airport_people_counter/app/detect_people.py:908`  
**Severity:** High — pipeline crashed with NameError on first line-crossing event

### Problem

When the first person crossed the counting line, the pipeline crashed with:

```
NameError: name 'recent_events' is not defined
```

### Root Cause

The variable `recent_events` was used inside the main loop (line 962: `recent_events.extend(snap["events_since_last"])`) and passed to `build_dashboard_model()` (line 1008), but it was never initialized before the `while True:` loop. The variable was only assigned later inside a conditional block, so on the first iteration when events existed, Python raised a `NameError`.

### Fix

Added initialization before the main loop:

```python
# detect_people.py, line 908 (before while True:)
recent_events = []
```

### Verification

Pipeline ran without crashing. Events accumulated correctly in the `recent_events` list and appeared in the dashboard's `EventTimeline` component.

---

## Fix 5: Duplicate Event IDs in Dashboard Response

**Date:** Integration session  
**File:** `airport_people_counter/app/snapshot.py:173`  
**Severity:** Medium — React console logged 120+ duplicate key warnings; events rendered incorrectly

### Problem

The browser console showed over 120 React duplicate key errors. The `EventTimeline` component rendered events incorrectly because every event in the response had `id: 0`.

### Root Cause

In `snapshot.py`, the `_events_to_api()` method iterated over events with:

```python
for e in events:
    ...
    out.append(Event(id=0, ...))
```

Every event was assigned `id=0` instead of a unique identifier. React's list rendering requires unique `key` props, so the duplicate IDs caused warnings and potential rendering bugs.

### Fix

Changed the loop to use `enumerate()` for sequential IDs:

```python
# Before
for e in events:
    ...
    out.append(Event(id=0, ...))

# After (line 173)
for idx, e in events:
    ...
    out.append(Event(id=idx, ...))
```

### Verification

Console duplicate key errors dropped to zero. Events in the `EventTimeline` component rendered correctly with unique IDs. The `EventInfo` interface in `api.ts` defines `id: number` which now receives unique sequential values.

---

## Fix 6: Camera Resolution Display

**Date:** Integration session  
**File:** Backend snapshot response (runtime behavior)  
**Severity:** Low — cosmetic; dashboard showed `?x?` instead of actual resolution

### Problem

The dashboard camera panel displayed `?x?` for resolution even though the camera was running at 1280x720.

### Root Cause

The `camera_state` object in the snapshot did not include the actual capture resolution. The fields `width` and `height` were not being populated from the stream's actual frame dimensions.

### Fix

The `SnapshotBuilder._camera_state()` method now reads the actual resolution from the stream metadata:

```python
camera_state=cls._camera_state(s, camera_config),
```

Where `_camera_state()` populates `width` and `height` from `stream.frame_w` / `stream.frame_h` (or the frame shape), ensuring the snapshot reflects the real capture resolution.

### Verification

Dashboard displayed `1280x720` in the camera panel, matching the actual webcam output resolution.

---

## Summary

| # | Fix | File | Severity | Impact |
|---|-----|------|----------|--------|
| 1 | API endpoint path | `api.ts:119` | Critical | Dashboard now receives live data |
| 2 | Empty chart guards | `charts.tsx` | Critical | Dashboard loads without crashing |
| 3 | History buffer write | `detect_people.py` | High | Trend charts populate |
| 4 | `recent_events` init | `detect_people.py:908` | High | Pipeline stable on first event |
| 5 | Event ID uniqueness | `snapshot.py:173` | Medium | No React key errors |
| 6 | Resolution display | `snapshot.py` | Low | Correct `1280x720` shown |

All fixes verified with:
- Backend: 169/169 unit tests passing (`make test`)
- Frontend: TypeScript clean (`tsc --noEmit`), build succeeds
- Integration: Dashboard shows live camera data, events, and charts with zero console errors

---

## Feature: Interactive Camera Line Mapping & Calibration

**Date:** Mapping/calibration session
**Files:** `app/queue_prototype.py`, `app/mapping.py` (new), `app/reload_signal.py` (new), `app/api_server.py`, `app/dashboard_models.py`, `app/detect_people.py`, `app/snapshot.py`, `config/mappings.yaml` (new), `config/camera.yaml`, `Queue-monitoring-dashboard/src/{lib/mapping.ts,components/MappingEditor.tsx,lib/api.ts,telemetry.ts,components/sections.tsx}`

### Problem

Geometry calibration (queue ROI, IN/OUT counting lines) required editing
`config/camera.yaml` source comments in pixel coordinates — error-prone,
camera/ops unreachable, and the legacy single counting line conflated the
IN and OUT semantics of two different events.

### Fix

1. **`config/mappings.yaml`** — per-camera, per-frame-normalised geometry
   (`roi` polygon + separate `in` / `out` lines) validated by
   `app/mapping.py.validate_mapping` and persisted **atomically**
   (serialise → re-parse → tempfile → `os.replace`) with automatic `.bak`
   of the previous store. A valid mapping is the single active geometry
   source; the legacy `camera.yaml` ROI/line is an in-memory fallback
   only, never combined.
2. **Crossing engine** — per-track/per-line state with a
   `crossing_debounce_frames` hold (default 2) and dedup; forward direction
   commits the counter change, the opposite direction emits an explicit
   `line_crossing_reversed` event that changes nothing. Only a forward OUT
   crossing finalises exit + waiting time; occupancy is live ROI membership
   using the configurable reference point (`bottom_center` default).
3. **API** — `GET`/`PUT /api/dashboard/mapping` (validate before write,
   failed updates leave the last known-good mapping active) and the raw
   `?annotated=0` stream for the editor. `GET /api/dashboard/config`
   gained additive mapping fields.
4. **Editor UI** — dashboard "Edit mapping" opens `MappingEditor.tsx`: draw
   and drag ROI + line handles over the raw letterboxed feed, keyboard
   nudging/swapping, validation feedback, revert, apply; a hot-reload
   signal switches the running detector to the new mapping on the next
   frame and resets per-track crossing state.

### Verification

- Backend: 349→350 pytest passing (mapping 25, mapping API 9, mapping-driven crossings 10 new)
- Frontend: 30 Vitest tests passing (`pnpm test`), `tsc --noEmit` clean, `pnpm run build` green
- Live end-to-end (video source + API + dashboard): editor drag & keyboard nudge → Save → `mappings.yaml` rewritten + `.bak` → `Config reloaded via api` in backend log; invalid PUT rejected with the disk untouched
- Both MJPEG variants streaming (annotated vs `?annotated=0`)
