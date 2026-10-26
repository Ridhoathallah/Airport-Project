# Queue Monitoring Dashboard — Frontend Quick Reference

> **Full documentation:** [../docs/](../docs/) — architecture
> ([ARCHITECTURE](../docs/ARCHITECTURE.md)), API contracts
> ([API](../docs/API.md)), dev guide ([DEVELOPMENT](../docs/DEVELOPMENT.md)).
> Agent instructions: [AGENTS.md](AGENTS.md) (this folder) and
> [../AGENTS.md](../AGENTS.md) (repo).

React 19 + TypeScript + Vite 8 + Tailwind CSS v4. **Presentation only** —
the Python backend computes every queue metric.

## Quick Start

Requires **Node.js `^20.19.0 || >=22.12.0`** — Vite 8's supported range
(this repo pins Node 22 in `.mise.toml`).

```bash
pnpm install        # first time (pnpm via corepack); npm also works
pnpm dev            # dev server :8443 (proxies /api → localhost:8080)
pnpm test           # vitest — 40 tests (as of 2026-10-23)
npx tsc --noEmit    # type check
pnpm run build      # production build to dist/
pnpm run format     # oxfmt
```

### Clean-clone prerequisite: `.figma/make/site.json`

`vite.config.ts` statically imports `.figma/make/site.json` (document-shell
metadata inherited from the Figma Make scaffold: title, description, robots,
icons — all optional). The `.figma/` directory is **gitignored on purpose**,
so a fresh clone has no such file and `pnpm dev` / `pnpm run build` fail at
config load until one exists. Create it before the first run with neutral
values — do not commit generated values:

```bash
mkdir -p .figma/make
echo '{}' > .figma/make/site.json   # every field is optional in vite.config.ts
```

Backend must be running: `./start.sh` from the repo root
(see [../README.md](../README.md)).

## Architecture

```
App.tsx — layout composition, theme toggle
├── QueueStatusPanel  — state chip, occupancy meter, trend/wait strip
├── CameraPanel       — MJPEG feed (annotated or raw in the editor)
│   └── MappingEditor — “Edit mapping”: ROI + IN/OUT lines over raw feed
├── MetricGrid        — “Key metrics”: entry/exit rate (sparklines), throughput, net change
├── AnalyticsPanel    — “Queue & traffic”: queue length + entries-vs-exits charts, range selector
├── WaitingTimePanel  — average / longest wait + wait trend chart
├── ActivityLog       — “Recent activity”: filter All / Passenger flow / Alerts
└── OperatorHealth    — health chip → DiagnosticsDrawer (pipeline diagnostics, raw snapshot)
```

### Data flow

- `useRealTelemetry()` polls `/api/dashboard/snapshot` @ 1 Hz via the
  Vite proxy → converts to `Snapshot`; connectivity (backend / camera /
  frames) is derived separately from the queue condition, which is always
  the backend classifier's own answer. CameraPanel plays
  `/api/camera/stream` (or `?annotated=0` inside the mapping editor);
  `useCameraConfig()` re-fetches config after a mapping Save.
- A measurement the backend reports as `null` stays `null` in the
  frontend and renders as “—”; there is no synthetic data source.

## Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Layout composition, theme toggle |
| `src/lib/telemetry.ts` | `useRealTelemetry` (poll + honest unavailable states), `useCameraConfig` |
| `src/lib/api.ts` | Typed client: `getSnapshot`, `getConfig`, `getHealth`, `getVideoStreamUrl(annotated)`, `getMapping`, `putMapping`, `shutdownApi` |
| `src/lib/mapping.ts` | Pure geometry transforms + validation (vitest-tested) |
| `src/lib/ui.tsx` | Formatters, colors, StatusChip, Panel primitives |
| `src/components/sections.tsx` | Dashboard panels incl. CameraPanel |
| `src/components/MappingEditor.tsx` | Interactive calibration dialog |
| `src/components/charts.tsx`, `health.tsx` | Charts, OperatorHealth, drawer |
| `vite.config.ts` | `/api` proxy (`API_PROXY_TARGET`), ports (`PORT`) |
| `vitest.config.ts` | Unit tests for pure logic |

## CameraPanel

MJPEG `<img>` with annotations burned in by the backend; **“Edit mapping”**
opens the calibration editor on the raw (`?annotated=0`) stream. When the
connection is not live the panel greys out and names the actual problem
(reconnecting, no frames, backend unreachable).

## Component Data Contract

All components receive a single `Snapshot` prop (from the backend):

```typescript
interface Snapshot {
  connection: "LIVE" | "DEGRADED" | "RECONNECTING" | "NO_FRAMES"
            | "CAMERA_OFFLINE" | "NO_DATA" | "BACKEND_UNAVAILABLE";
  state: "EMPTY" | "SPARSE" | "NORMAL" | "CROWDED" | "CONGESTED"
       | "BACKLOG" | "UNCONFIGURED" | "DATA_UNAVAILABLE";  // backend classifier, pass-through
  stateActive: boolean;
  reason: string;
  queue: number | null; capacity: number; occupancy: number | null;
  congestionSec: number | null;
  entryRate: number | null; exitRate: number | null; netChange: number | null;
  trend: "GROWING" | "CLEARING" | "STEADY" | null;
  avgWait: number | null; longestWait: number | null; throughput: number | null;
  camera: CameraHealth; cv: CvHealth;
  recording: { active: boolean; sinceSec: number | null; sizeGb: number | null };
  lastKnownQueue: number | null; lastUpdate: number;
  history: Sample[];      // ≤240 points (60 min buffer @1 Hz)
  events: QueueEvent[]; tracks: Track[]; frameTs: number;
}
```

Wire-level types match `../docs/API.md`; frontend never computes queue
metrics itself.

## Common Issues

| Symptom | Fix |
|---------|-----|
| "API error" | Backend not running (`./start.sh` or `./run.sh --headless --api`) |
| MJPEG not loading | Open `/api/camera/stream` directly; backend on 8080; image sources are single-shot (use camera/video) |
| 502 from proxy | Backend down, or wrong `API_PROXY_TARGET` |
| “Backend unreachable” chip | Backend down or unreachable; the last reading stays on screen marked stale |
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
