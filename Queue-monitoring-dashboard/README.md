# Queue Monitoring Dashboard — Frontend Quick Reference

> **Full documentation:** [../docs/](../docs/)

## Quick Start

```bash
# Install dependencies (first time)
npm install

# Development server (port 8443)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Format code
npm run format
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DASHBOARD                                 │
├─────────────────────────────────────────────────────────────────┤
│  App.tsx                                                        │
│  ├── Source Selector: "Simulator" │ "Live API"                 │
│  ├── QueueStatusPanel  ── State chip, occupancy, trend/wait    │
│  ├── CameraPanel       ── MJPEG stream + YOLO/SORT overlays    │
│  ├── KpiStack          ── 9 metric tiles with sparklines       │
│  ├── AnalyticsPanel    ── FlowChart + OccupancyChart           │
│  ├── EventTimeline     ── Filterable event log (All/Flow/Alerts)│
│  └── DiagnosticsDrawer ── Raw snapshot + scenario selector     │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Simulator Mode (Default)
```
useTelemetry(scenario) → Synthetic queue dynamics (1 Hz tick)
                    → Internal Snapshot type
                    → Components render
```

### Live API Mode
```
useRealTelemetry() → Polls /api/snapshot @ 1 Hz (via Vite proxy)
                → Converts API response → Internal Snapshot type
                → Components render
                → CameraPanel loads MJPEG from /api/camera/stream
```

## Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Root: source selector, layout composition |
| `src/lib/telemetry.ts` | `useTelemetry` (sim) + `useRealTelemetry` (API) |
| `src/lib/api.ts` | Typed fetch: `getSnapshot()`, `getConfig()`, `getHealth()`, `getVideoStreamUrl()` |
| `src/lib/ui.ts` | Formatters, colors, StatusChip, Panel components |
| `src/components/sections.tsx` | All dashboard panels + CameraPanel |
| `src/components/charts.tsx` | Sparkline, OccupancyChart, FlowChart |
| `src/components/health.tsx` | OperatorHealth, DiagnosticsDrawer |
| `vite.config.ts` | Vite config + `/api` proxy → `localhost:8080` |

## Source Selector (Top Bar)

| Source | Description |
|--------|-------------|
| **Simulator** | Synthetic data — choose scenario: NORMAL/BUSY/CONGESTED/BACKLOG/OFFLINE/DEGRADED |
| **Live API** | Real backend data — requires `./run.sh --api` in backend |

## CameraPanel

| Mode | Behavior |
|------|----------|
| **Simulator** | Fake CCTV scene (gradient + grid) with SVG overlays |
| **Live API** | MJPEG `<img>` stream + HTML overlays (boxes, IDs, ROI, line) |

**Overlays (toggleable):**
- People boxes (with IDs)
- Queue area (ROI polygon)
- Counting line (ENTRY/EXIT)
- Box colors: Green = in ROI, Grey = outside ROI

## API Integration

### Vite Proxy
```typescript
// vite.config.ts
proxy: {
  '/api': {
    target: 'http://localhost:8080',
    changeOrigin: true,
    rewrite: (path) => path,
  },
}
```

### TypeScript Client (`src/lib/api.ts`)
```typescript
// Polling
const snapshot = await getSnapshot();     // GET /api/snapshot
const config = await getConfig();         // GET /api/config
const health = await getHealth();         // GET /api/health

// MJPEG stream URL
<img src={getVideoStreamUrl()} />         // /api/camera/stream
```

### React Hook (`src/lib/telemetry.ts`)
```typescript
const { snap, error, refetch } = useRealTelemetry(1000); // 1 Hz poll
// Returns internal Snapshot type (converted from API response)
```

## Component Data Contract

All components receive a single `Snapshot` prop (from either simulator or API):

```typescript
interface Snapshot {
  scenario: "NORMAL" | "BUSY" | "CONGESTED" | "BACKLOG" | "OFFLINE" | "DEGRADED";
  state: "NORMAL" | "BUSY" | "CONGESTED" | "BACKLOG";
  stateActive: boolean;
  reason: string;
  queue: number;
  capacity: number;
  occupancy: number;
  congestionSec: number;
  entryRate: number;
  exitRate: number;
  netChange: number;
  trend: "GROWING" | "CLEARING" | "STEADY";
  avgWait: number;
  longestWait: number;
  throughput: number;
  camera: CameraHealth;
  cv: CvHealth;
  recording: { active: boolean; sinceSec: number; sizeGb: number };
  lastKnownQueue: number;
  lastUpdate: number;
  history: Sample[];
  events: QueueEvent[];
  tracks: Track[];
  frameTs: number;
}
```

## Testing

```bash
# Type check
npx tsc --noEmit

# Lint/format
npm run format

# Build verification
npm run build
```

## Common Issues

| Symptom | Fix |
|---------|-----|
| "API error" in Live API | Backend not running: `./run.sh --api` |
| MJPEG not loading | Check `/api/camera/stream` in browser; verify backend on 8080 |
| CORS errors | Vite proxy must be running (`npm run dev`) |
| Dashboard shows "OFFLINE" | Source selector = "Live API" but backend not running |
| Simulator scenarios don't change | Check DiagnosticsDrawer → scenario selector |

## Ports

| Service | Port |
|---------|------|
| Vite Dev Server | 8443 |
| FastAPI Backend | 8080 (proxied via `/api`) |

## Dependencies

- React 19 + React DOM 19
- Vite 8 + TypeScript 5.7
- Tailwind CSS v4
- @figma/astraui + @figma/astraui-kit
- lucide-react (icons)

---

**Last reviewed against code:** 2026-09-02