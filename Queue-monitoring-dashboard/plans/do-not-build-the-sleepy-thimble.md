# Design Plan — Airport X-Ray Queue Monitoring Dashboard

## Context

We are building the operator-facing UI for a **camera-only** computer-vision prototype that
monitors a single airport security screening lane. The pipeline is:

`Camera → YOLOv5 person detection → SORT tracking → ROI/line analytics → occupancy → entry/exit flow → waiting time → congestion classification → event logging`

The dashboard is for **airport/security operations staff doing continuous real-time monitoring**.
The single most important design goal: an operator glancing at the screen must answer the core
situational questions in **3–5 seconds** — how many are waiting, is it normal/busy/congested,
is it growing or clearing, how long is the wait, and is the camera + CV pipeline healthy.

Scope discipline (hard constraints):
- **One camera only.** Layout may leave room for future multi-camera, but the UI must NOT imply
  MQTT, cloud processing, physical sensors, facial recognition, passenger identity, or sensor fusion
  exist today.
- Anonymous tracking IDs only — no identity.
- Optimized for **desktop 1440×900**, dark control-room aesthetic.

This document is the design/IA plan only. **No final UI is built yet.**

---

## 1. Recommended Page / Layout Structure

Single-screen operational console (no scrolling required for the critical picture at 1440×900).
A persistent top status bar + a fixed 3-column working grid + a bottom event strip.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ TOP BAR: lane name · system clock/uptime · GLOBAL STATUS pill · camera+CV   │
│          health chips · recording indicator                                 │
├──────────────────────────────┬───────────────────────┬─────────────────────┤
│                              │  QUEUE STATUS PANEL    │  KPI STACK          │
│   LIVE CAMERA VIEW           │  (big state, people,   │  (8 core metrics as │
│   (CCTV, bboxes, IDs, ROI,   │   occupancy %, capacity│   compact stat      │
│    entry/exit line, FPS)     │   bar, congestion dur, │   tiles, 2 cols)    │
│   ~55% width, tallest cell   │   reason)              │                     │
│                              ├───────────────────────┴─────────────────────┤
│                              │  TRAFFIC / TREND CHARTS                       │
│                              │  (entries vs exits · queue trend ·           │
│                              │   waiting/occupancy) w/ 5m·15m·30m·60m toggle │
├──────────────────────────────┴──────────────────────────────────────────────┤
│ EVENT TIMELINE (scrolling log, severity-coded, newest first)                 │
└───────────────────────────────────────────────────────────────────────────┘
```

Right rail (health detail for camera + CV) lives in a collapsible drawer / secondary panel so it
is available but not stealing the primary glance-zone. Health *summary* chips stay in the top bar.

---

## 2. Information Hierarchy (glance order)

1. **Global queue state** — NORMAL / BUSY / CONGESTED / BACKLOG (largest, color + text + icon).
2. **People in queue** + occupancy % + capacity — the single number an operator wants most.
3. **Trend direction** — growing / clearing (net queue change, arrow + sparkline).
4. **Live camera** — visual confirmation the numbers are real; boxes/IDs/ROI/line.
5. **Wait times + flow rates** — avg/longest wait, entry/exit rate, throughput.
6. **System trust** — camera health + CV pipeline health (must be reassuring at a glance).
7. **Recent events** — timeline for "what just happened / why."

Design rule: **color never carries meaning alone** — every status has a text label + icon/shape.

---

## 3. Main Dashboard Sections

1. **Top status bar** — lane id, clock, uptime, global status pill, camera-health chip, CV-health
   chip, recording dot. This row alone should answer "is everything OK?"
2. **Live camera view** — CCTV frame, person bounding boxes, anonymous track IDs, ROI/queue polygon
   overlay, entry/exit line, live FPS + detection/track counts, timestamp burn-in.
3. **Queue status panel** — dominant state indicator, people-in-queue, occupancy % with capacity
   bar, current capacity, congestion duration timer, and machine-readable "reason" (e.g. "entry
   rate > exit rate for 3m", "occupancy > 85%").
4. **Core KPI stack** — 8 tiles: people in queue, occupancy %, entry rate (/min), exit rate (/min),
   avg wait, longest wait, throughput (/hr), net queue change. Each tile: value, unit, tiny
   sparkline, delta vs prior window.
5. **Traffic analytics** — entries vs exits (dual series), queue-size trend, with 5m/15m/30m/60m
   range selector shared across charts.
6. **Waiting & occupancy trends** — wait-time line + occupancy area over selected range.
7. **Event timeline** — entries/exits (throttled/aggregated), state changes, congestion
   started/ended, system events (camera reconnect, CV degradation). Severity-coded, filterable.
8. **Camera health (detail)** — connection, FPS, resolution, latency, packet loss, last frame age,
   uptime.
9. **CV pipeline health (detail)** — detection status, tracking status, analytics status, inference
   time (ms), model (YOLOv5) + tracker (SORT) info.
10. **Recording status** — state, duration, storage indicator.

---

## 4. Recommended Component Structure

```
App.tsx
└─ DashboardLayout
   ├─ TopStatusBar
   │    ├─ LaneIdentity        (lane name + clock + uptime)
   │    ├─ GlobalStatusPill    (NORMAL/BUSY/CONGESTED/BACKLOG)
   │    ├─ HealthChip          (reused: camera + CV summary)
   │    └─ RecordingIndicator
   ├─ CameraPanel
   │    ├─ CameraCanvas        (frame + overlay layers)
   │    ├─ DetectionOverlay    (bboxes, track IDs)
   │    ├─ RoiLineOverlay      (ROI polygon + entry/exit line)
   │    └─ CameraHud           (FPS, det/track counts, timestamp)
   ├─ QueueStatusPanel
   │    ├─ StateBadge          (large, reused StatusToken)
   │    ├─ OccupancyMeter      (bar w/ capacity + threshold marks)
   │    └─ CongestionReason    (duration timer + reason text)
   ├─ KpiStack
   │    └─ KpiTile ×8          (value, unit, sparkline, delta)
   ├─ AnalyticsPanel
   │    ├─ RangeSelector       (5m/15m/30m/60m, shared state)
   │    ├─ FlowChart           (entries vs exits)
   │    ├─ QueueTrendChart
   │    └─ WaitOccupancyChart
   ├─ EventTimeline
   │    ├─ TimelineFilters
   │    └─ EventRow ×N         (severity-coded, reused StatusToken)
   └─ HealthDrawer
        ├─ CameraHealthGrid
        └─ CvPipelineHealthGrid
```

Shared primitives to build once and reuse: `StatusToken` (color+icon+label mapping for the 4 queue
states + health states), `Panel` (consistent header/border/padding), `MetricValue` (number +
unit + delta formatting), `Sparkline`, `Meter`.

Data layer: a single mock `useQueueTelemetry()` hook produces the live-updating fixture stream
(occupancy, rates, events, health, frames) so every component reads from one source and the
"live" feel is coherent. Keeps the prototype honest about being camera-only.

---

## 5. Above the Fold (1440×900, no scroll)

- Top status bar (global status + health chips + recording).
- Live camera view.
- Queue status panel (state, people, occupancy, capacity, congestion duration/reason).
- All 8 core KPI tiles.
- Traffic/trend charts with range selector.
- First rows of the event timeline.

Below fold / on-demand: full camera-health + CV-pipeline detail grids (drawer), extended event
history, chart deep-dives.

---

## 6. Recommended Chart Types

- **Entries vs exits:** dual-line (or paired mini-bars per bucket) — two series, shared x-axis time.
- **Queue trend / net change:** area or line with a zero/growing-vs-clearing emphasis.
- **Waiting time:** line chart (avg) with a band or second series for longest.
- **Occupancy:** area chart with capacity threshold lines (busy/congested marks).
- **KPI tiles:** sparklines (compact, axis-free) for micro-trend.
- **Occupancy meter:** horizontal bar with threshold ticks, not a gauge/donut.
- Avoid pie/donut, 3D, and decorative gauges. Follow the `dataviz` skill for palette + a single
  coherent chart system across light/dark.

---

## 7. Important Interaction States

- **Range selector** (5m/15m/30m/60m) drives all analytics charts consistently.
- **Hover tooltips** on charts + sparklines with exact value + timestamp.
- **Event filtering** by type/severity; click event → highlight corresponding time on charts (nice-to-have).
- **Camera overlay toggles** — show/hide bboxes, IDs, ROI, entry/exit line (operator control).
- **Health drawer** open/close.
- **Loading / no-data / stale-data** states for charts and camera (e.g. "last frame 4s ago").
- **Threshold crossing** animation restraint — subtle pulse on the global status pill when state
  changes; no distracting motion during steady monitoring.

---

## 8. Status / Condition States (whole-dashboard behavior)

- **NORMAL (green):** calm chrome, green status pill, steady metrics. Baseline.
- **BUSY (amber):** amber pill + "BUSY" label, occupancy meter in amber zone, KPI deltas
  emphasized; no alarm.
- **CONGESTED / CRITICAL (orange→red):** red pill, congestion duration timer running, reason shown,
  occupancy meter in red zone, top bar accent. Elevated but non-panic.
- **BACKLOG:** distinct from congested — sustained overflow; emphasize net-change + longest wait.
- **OFFLINE / DEGRADED:** camera disconnected or CV stalled → camera panel shows offline overlay
  ("SIGNAL LOST · last frame HH:MM:SS"), health chips go red, analytics show stale badge, global
  status shows "MONITORING UNAVAILABLE" rather than a false NORMAL. This is critical — never imply
  a healthy queue when the pipeline is blind.

Each state defined by tokens (bg/border/text/accent) driven off a single status enum.

---

## 9. Responsive Strategy

- **Primary target: 1440×900 desktop** — fixed 3-column console, no scroll for critical zone.
- **≥1920:** widen camera + charts, allow multi-camera grid slot to appear (structurally reserved,
  shows single camera today).
- **1280–1439:** compress KPI tiles to smaller type, keep 3 columns.
- **<1280 / tablet:** collapse to stacked single column (camera → status → KPIs → charts → events),
  health moves fully into drawer. Not a primary use case but must not break.
- Not designed for mobile phones (control-room product).

---

## 10. Overall Visual Direction

- **Dark control-room / SCADA / NOC aesthetic.** Near-black desaturated background (deep
  blue-gray, not pure black), elevated panels via subtle borders + minimal elevation — not
  glassmorphism, not gradients.
- **Information-dense, gridded, aligned.** Tight but breathable spacing; every panel same header
  treatment. Calm by default so status color reads instantly when it appears.
- **Typography:** a technical/geometric sans for UI (e.g. Inter / IBM Plex Sans) + a **monospace or
  tabular-figure** face for all numeric metrics, IDs, timestamps, and FPS so values don't jitter as
  they update. (Final pairing chosen via the `aesthetic-stance` skill at build time.)
- **Color = status only.** Green/amber/orange-red/blue reserved for meaning; UI chrome stays
  neutral. Always pair color with label + icon.
- **Credible instrumentation cues:** live timestamps, FPS, inference-ms, packet loss, uptime — the
  details that make an operator trust the system.
- Avoid: generic SaaS look, decorative illustration, oversized hero cards, excessive gradients,
  consumer polish.

---

## Build Approach (when approved)

1. Invoke `aesthetic-stance` (mandatory — no Make Kit) to commit font pairing + design tokens, and
   call `create_make_theme` for the page-level dark control-room direction. Follow `dataviz` before
   any chart.
2. Establish tokens + global styles in `src/index.css` (dark theme, status color scale, tabular
   number font wiring).
3. Build shared primitives (`StatusToken`, `Panel`, `MetricValue`, `Meter`, `Sparkline`).
4. Build `useQueueTelemetry()` mock stream (single source of truth, camera-only fixtures).
5. Compose sections top-down in `src/App.tsx` → `DashboardLayout` and children.
6. Wire status enum → all state-dependent styling; implement Normal/Busy/Congested/Backlog/Offline.
7. Charts (Recharts or equivalent) with shared range selector.

## Verification

- Load preview at 1440×900; confirm the critical zone fits with no scroll and the 3–5s glance test
  passes (state, count, trend, health all visible immediately).
- Cycle mock stream through Normal → Busy → Congested → Backlog → Offline; confirm color + text +
  icon all change and Offline never shows a false-healthy queue.
- Confirm no references to MQTT/cloud/sensors/identity anywhere in UI copy.
- Check numeric readouts use tabular/mono figures (no layout jitter on update).
```
