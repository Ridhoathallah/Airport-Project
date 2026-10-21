import { useEffect, useState } from "react";
import { useTelemetry, useRealTelemetry, useCameraConfig, type Scenario } from "./lib/telemetry";
import { shutdownApi } from "./lib/api";
import {
  QueueStatusPanel,
  CameraPanel,
  MetricGrid,
  AnalyticsPanel,
  WaitingTimePanel,
  ActivityLog,
} from "./components/sections";
import { OperatorHealth, DiagnosticsDrawer } from "./components/health";

export type AppMode = "live" | "demo";

/**
 * Airport X-Ray Queue Monitoring System
 *
 * A single-screen control room. The queue status panel is the focal point
 * and answers the operator's first question, the camera feed supports it,
 * and the right column holds the flow, trend and waiting time detail.
 *
 * At 1440x900 and above the console fills the viewport without scrolling.
 * Below that the panels stack and the page scrolls, which keeps every
 * reading reachable instead of squeezing the controls off screen.
 */
export default function App() {
  const [mode, setMode] = useState<AppMode>("live");
  const [scenario, setScenario] = useState<Scenario>("NORMAL");
  const [drawer, setDrawer] = useState(false);
  const [range, setRange] = useState("15m");
  const [clock, setClock] = useState(Date.now());

  const simSnap = useTelemetry(scenario);
  const { snap: apiSnap, error: apiError, refetch } = useRealTelemetry();
  const [camConfig, refreshCamConfig] = useCameraConfig();

  const isLive = mode === "live";
  const s = isLive ? apiSnap : simSnap;
  const dataSource = isLive ? "api" : "simulator";

  useEffect(() => {
    const id = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  if (!s) return <div className="size-full bg-brand-tertiary" />;

  const onShutdown = () => {
    if (window.confirm("Shut down the detection pipeline and API server?")) {
      shutdownApi().catch(() => {});
    }
  };

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-brand-tertiary text-text-primary">
      {/* ── Status bar ─────────────────────────────────────────────────────
          Identity on the left, the four conditions that decide whether a
          reading can be trusted on the right, the clock last. */}
      <header className="flex h-14 shrink-0 items-center justify-between gap-md border-b border-border-primary bg-surface-bg px-lg">
        <div className="flex min-w-0 items-center gap-md">
          <div className="flex min-w-0 flex-col">
            <h1 className="truncate text-[15px] font-semibold leading-tight tracking-tight text-text-primary">
              Airport X-Ray Queue Monitoring System
            </h1>
            <span className="truncate text-[11px] leading-tight text-text-tertiary">
              Terminal 1 · Security Screening
            </span>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-md">
          {isLive && apiError && (
            <div className="flex items-center gap-xs rounded-corner-sm border border-border-control px-sm py-1">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--st-busy)" }} aria-hidden />
              <span className="text-[11px] font-semibold" style={{ color: "var(--st-busy)" }}>
                Reconnecting
              </span>
              <button
                type="button"
                onClick={refetch}
                className="text-[11px] text-text-tertiary transition-colors hover:text-text-primary"
                title="Try to reconnect"
                aria-label="Try to reconnect"
              >
                Retry
              </button>
            </div>
          )}

          <div className="hidden xl:block">
            <OperatorHealth s={s} onOpen={() => setDrawer(true)} />
          </div>
          <div className="xl:hidden">
            <OperatorHealth s={s} onOpen={() => setDrawer(true)} compact />
          </div>

          <time className="tnum pl-xs text-[15px] font-semibold leading-none text-text-primary">
            {new Date(clock).toLocaleTimeString("en-GB", { hour12: false })}
          </time>
        </div>
      </header>

      {/* ── Working grid ─────────────────────────────────────────────────
          Left column: the condition and the camera. Right column: the flow,
          the trend and the waiting time. */}
      <main className="grid min-h-0 flex-1 grid-cols-1 gap-md overflow-y-auto p-md xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] xl:overflow-hidden">
        {/* The columns keep their content height below the two-column
            breakpoint. `min-h-0` would let the grid rows collapse, and the
            left column would overflow into the right one. */}
        <div className="flex flex-col gap-md xl:min-h-0">
          <QueueStatusPanel s={s} />
          <CameraPanel s={s} dataSource={dataSource} config={camConfig} onConfigChanged={refreshCamConfig} />
        </div>

        <div className="flex flex-col gap-md xl:min-h-0">
          <MetricGrid s={s} />
          <AnalyticsPanel s={s} range={range} onRange={setRange} />
          <WaitingTimePanel s={s} range={range} config={camConfig} />
        </div>
      </main>

      {/* ── Recent activity ───────────────────────────────────────────────
          A fixed strip so the log never pushes the monitoring panels out of
          view. */}
      <div className="h-[140px] shrink-0 px-md pb-md sm:h-[164px]">
        <ActivityLog s={s} />
      </div>

      <DiagnosticsDrawer
        s={s}
        open={drawer}
        onClose={() => setDrawer(false)}
        config={camConfig}
        dataSource={dataSource}
        onDataSource={(ds) => setMode(ds === "api" ? "live" : "demo")}
        scenario={scenario}
        onScenario={setScenario}
        onShutdown={onShutdown}
      />
    </div>
  );
}
