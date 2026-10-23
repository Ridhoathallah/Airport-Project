# Queue Monitoring Dashboard — Frontend Quick Reference

> **Full documentation:** [../docs/](../docs/) — architecture
> ([ARCHITECTURE](../docs/ARCHITECTURE.md)), API contracts
> ([API](../docs/API.md)), dev guide ([DEVELOPMENT](../docs/DEVELOPMENT.md)).
> Agent instructions: [AGENTS.md](AGENTS.md) (this folder) and
> [../AGENTS.md](../AGENTS.md) (repo).

React 19 + TypeScript + Vite 8 + Tailwind CSS v4. **Presentation only** —
the Python backend computes every queue metric.

## Quick Start

```bash
pnpm install        # first time (pnpm via corepack); npm also works
pnpm dev            # dev server :8443 (proxies /api → localhost:8080)
pnpm test           # vitest — 30 tests (as of 2026-10-21)
npx tsc --noEmit    # type check
pnpm run build      # production build to dist/
pnpm run format     # oxfmt
```

Backend must be running for LIVE mode: `./start.sh` from the repo root
(see [../README.md](../README.md)).

## Architecture

```
App.tsx — mode switch: LIVE (API) | DEMO (simulator), layout composition
├── QueueStatusPanel  — state chip, occupancy meter, trend/wait strip
├── CameraPanel       — MJPEG feed (LIVE) / simulated scene (DEMO)
│   └── MappingEditor — “Edit mapping”: ROI + IN/OUT lines over raw feed
├── MetricGrid        — “Key metrics”: entry/exit rate (sparklines), throughput, net change
├── AnalyticsPanel    — “Queue & traffic”: queue length + entries-vs-exits charts, range selector
├── WaitingTimePanel  — average / longest wait + wait trend chart
├── ActivityLog       — “Recent activity”: filter All / Passenger flow / Alerts
└── OperatorHealth    — health chip → DiagnosticsDrawer (raw snapshot, scenario selector)
```

### Data flow

- **DEMO:** `useTelemetry(scenario)` → synthetic 1 Hz dynamics → `Snapshot`
- **LIVE:** `useRealTelemetry()` polls `/api/dashboard/snapshot` @ 1 Hz via the
  Vite proxy → converts to the same `Snapshot`; CameraPanel plays
  `/api/camera/stream` (or `?annotated=0` inside the mapping editor);
  `useCameraConfig()` re-fetches config after a mapping Save.

## Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Mode switch, layout composition |
| `src/lib/telemetry.ts` | `useTelemetry` (sim), `useRealTelemetry`, `useCameraConfig` |
| `src/lib/api.ts` | Typed client: `getSnapshot`, `getConfig`, `getHealth`, `getVideoStreamUrl(annotated)`, `getMapping`, `putMapping`, `shutdownApi` |
| `src/lib/mapping.ts` | Pure geometry transforms + validation (vitest-tested) |
| `src/lib/ui.tsx` | Formatters, colors, StatusChip, Panel primitives |
| `src/components/sections.tsx` | Dashboard panels incl. CameraPanel |
| `src/components/MappingEditor.tsx` | Interactive calibration dialog |
| `src/components/charts.tsx`, `health.tsx` | Charts, OperatorHealth, drawer |
| `vite.config.ts` | `/api` proxy (`API_PROXY_TARGET`), ports (`PORT`) |
| `vitest.config.ts` | Unit tests for pure logic |

## CameraPanel

| Mode | Behaviour |
|------|-----------|
| **DEMO** | Simulated CCTV scene (SVG overlays: boxes/IDs/ROI/line, toggleable) |
| **LIVE** | MJPEG `<img>` with annotations burned in by the backend; **“Edit mapping”** opens the calibration editor on the raw (`?annotated=0`) stream |

## Component Data Contract

All components receive a single `Snapshot` prop (from either source):

```typescript
interface Snapshot {
  scenario: "NORMAL" | "BUSY" | "CONGESTED" | "BACKLOG" | "OFFLINE" | "DEGRADED";
  state: "NORMAL" | "BUSY" | "CONGESTED" | "BACKLOG";
  stateActive: boolean;
  reason: string;
  queue: number; capacity: number; occupancy: number;
  congestionSec: number;
  entryRate: number; exitRate: number; netChange: number;
  trend: "GROWING" | "CLEARING" | "STEADY";
  avgWait: number; longestWait: number; throughput: number;
  camera: CameraHealth; cv: CvHealth;
  recording: { active: boolean; sinceSec: number; sizeGb: number };
  lastKnownQueue: number; lastUpdate: number;
  history: Sample[];      // ≤240 points (60 min buffer @1 Hz)
  events: QueueEvent[]; tracks: Track[]; frameTs: number;
}
```

Wire-level types match `../docs/API.md`; frontend never computes queue
metrics itself.

## Common Issues

| Symptom | Fix |
|---------|-----|
| "API error" in LIVE | Backend not running (`./start.sh` or `./run.sh --headless --api`) |
| MJPEG not loading | Open `/api/camera/stream` directly; backend on 8080; image sources are single-shot (use camera/video) |
| 502 from proxy | Backend down, or wrong `API_PROXY_TARGET` |
| Dashboard OFFLINE | Source selector = LIVE but backend down |
| Buttons greyed in editor | Not dirty yet — move a handle first; Save appears after an edit |

## Ports

| Service | Port |
|---------|------|
| Vite dev server | `8443` (`PORT` env) |
| FastAPI backend | `8080` (proxied via `/api`) |

## Dependencies

React 19 · Vite 8 · TypeScript 5.7 · Tailwind CSS v4 · Vitest 5 ·
@figma/astraui (+ kit) · lucide-react

---

**Last reviewed against code:** 2026-10-21
