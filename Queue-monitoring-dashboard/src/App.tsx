import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";
import { useRealTelemetry, useCameraConfig } from "./lib/telemetry";
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

/**
 * Airport X-Ray Queue Monitoring System
 *
 * A single-screen control room. The queue status panel is the focal point
 * and answers the operator's first question, the camera feed supports it,
 * and the right column holds the flow, trend and waiting time detail.
 *
 * All telemetry comes from the backend; the console only presents it.
 * Connectivity (backend reachable, camera delivering) is tracked separately
 * from the queue condition, which is always the classifier's own answer.
 *
 * At 1440x900 and above the console fills the viewport without scrolling.
 * Below that the panels stack and the page scrolls, which keeps every
 * reading reachable instead of squeezing the controls off screen.
 */
export default function App() {
  const [drawer, setDrawer] = useState(false);
  const [range, setRange] = useState("15m");
  const [clock, setClock] = useState(Date.now());
  // Dark is the default; the inline script in index.html has already applied
  // a persisted light choice before this renders.
  const [theme, setTheme] = useState<"dark" | "light">(() =>
    document.documentElement.classList.contains("light") ? "light" : "dark",
  );

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.classList.toggle("light", next === "light");
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* storage unavailable: the theme still applies for this session */
    }
    setTheme(next);
  };

  const { snap: s, error: apiError, refetch } = useRealTelemetry();
  const [camConfig, refreshCamConfig] = useCameraConfig();

  useEffect(() => {
    const id = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

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
          <button
            type="button"
            className="ctl h-7 w-7 justify-center px-0"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
          </button>
          {apiError && (
            <div className="flex items-center gap-xs rounded-corner-sm border border-control px-sm py-1">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--st-busy)" }} aria-hidden />
              <span className="text-[11px] font-semibold" style={{ color: "var(--st-busy)" }}>
                Backend unreachable
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
          <CameraPanel s={s} config={camConfig} onConfigChanged={refreshCamConfig} />
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
        onShutdown={onShutdown}
      />
    </div>
  );
}
