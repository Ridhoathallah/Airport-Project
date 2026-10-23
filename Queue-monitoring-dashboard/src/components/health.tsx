import { useCallback, useEffect, useRef, useState } from "react";
import { X, Camera, Activity, Sliders, ShieldCheck, Terminal, Power } from "lucide-react";
import type { Snapshot, Scenario } from "../lib/telemetry";
import type { CameraSourceInfo, ConfigResponse } from "../lib/api";
import { getCameraSources, selectCameraSource, setCameraRotation } from "../lib/api";
import { StatusMark, healthStyle, QUEUE_STYLE, fmtDuration, fmtClock, type StatusStyle } from "../lib/ui";

// ─────────────────────────────────────────────────────────────────────────────
// Operator health: the four conditions that decide whether a reading can be
// trusted. Technical detail belongs in the diagnostics drawer, not here.
// ─────────────────────────────────────────────────────────────────────────────

function monitoringStyle(s: Snapshot): StatusStyle {
  if (!s.stateActive) return { color: "var(--st-backlog)", label: "Stopped", glyph: "square" };
  if (s.cv.detection === "DEGRADED") return { color: "var(--st-busy)", label: "Reduced", glyph: "diamond" };
  return { color: "var(--st-normal)", label: "Active", glyph: "circle" };
}

function dataStyle(s: Snapshot): StatusStyle {
  if (!s.stateActive) return { color: "var(--st-backlog)", label: "Stale", glyph: "square" };
  if (s.camera.link === "DEGRADED") return { color: "var(--st-busy)", label: "Delayed", glyph: "diamond" };
  return { color: "var(--st-normal)", label: "Updating", glyph: "circle" };
}

function recStyle(s: Snapshot): StatusStyle {
  return s.recording.active
    ? { color: "var(--st-normal)", label: "Active", glyph: "circle" }
    : { color: "var(--st-offline)", label: "Off", glyph: "ring" };
}

function HealthItem({ label, status }: { label: string; status: StatusStyle }) {
  return (
    <span className="flex items-center gap-xs">
      <span className="t-label text-text-tertiary">{label}</span>
      <span className="flex items-center gap-xs text-[11px] font-semibold" style={{ color: status.color }}>
        <StatusMark status={status} size={10} />
        {status.label}
      </span>
    </span>
  );
}

export function OperatorHealth({
  s,
  onOpen,
  compact = false,
}: {
  s: Snapshot;
  onOpen: () => void;
  compact?: boolean;
}) {
  if (compact) {
    // Narrow screens cannot fit four conditions. Show the one that decides
    // whether the reading can be trusted and keep the rest in diagnostics.
    const camera = healthStyle(s.camera.link);
    return (
      <div className="flex items-center gap-sm">
        <HealthItem label="Camera" status={camera} />
        <button
          type="button"
          className="ctl h-7 w-7 justify-center px-0"
          onClick={onOpen}
          aria-label="Open system diagnostics"
          title="System diagnostics"
        >
          <Sliders size={13} />
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-md">
      <HealthItem label="Camera" status={healthStyle(s.camera.link)} />
      <span className="h-4 w-px bg-rule" aria-hidden />
      <HealthItem label="Monitoring" status={monitoringStyle(s)} />
      <span className="h-4 w-px bg-rule" aria-hidden />
      <HealthItem label="Recording" status={recStyle(s)} />
      <span className="h-4 w-px bg-rule" aria-hidden />
      <HealthItem label="Data" status={dataStyle(s)} />
      <button
        type="button"
        className="ctl h-7 px-sm"
        onClick={onOpen}
        aria-label="Open system diagnostics"
      >
        <Sliders size={13} />
        Diagnostics
      </button>
    </div>
  );
}

/**
 * Age of the newest captured frame.
 *
 * The backend reports the real age of the last frame, so this shows the
 * measured value rather than "now" derived from the snapshot timestamp.
 */
function lastFrameLabel(s: Snapshot): string {
  const age = s.camera.lastFrameAgeMs;
  if (!s.stateActive || age >= 999999) return "No signal";
  if (age < 1000) return `${age} ms ago`;
  return `${(age / 1000).toFixed(1)} s ago`;
}

/** Measurement mode, taken from the API instead of assumed. */
function calibrationLabel(config: ConfigResponse | null | undefined): string {
  if (!config) return "Not reported";
  if (config.calibration === "METRES") return "Metres (calibrated)";
  return "Pixel relative (uncalibrated)";
}

function Row({ label, value, state }: { label: string; value?: string; state?: StatusStyle }) {
  return (
    <div className="flex items-center justify-between gap-md border-b border-border-primary py-xs last:border-0">
      <span className="text-[12px] text-text-secondary">{label}</span>
      {state ? (
        <span className="flex items-center gap-xs text-[12px] font-semibold" style={{ color: state.color }}>
          <StatusMark status={state} size={10} />
          {state.label}
        </span>
      ) : (
        <span className="t-meta text-text-primary">{value}</span>
      )}
    </div>
  );
}

const SCENARIOS: Scenario[] = ["NORMAL", "BUSY", "CONGESTED", "BACKLOG", "DEGRADED", "OFFLINE"];

const ROTATION_OPTIONS = [0, 90, 180, 270] as const;
const CAMERA_SOURCES_COUNT = 5;

/**
 * Which webcam feeds the lane and how the captured image is oriented.
 *
 * Set rarely (after a camera move or remount), so the controls live in
 * diagnostics, not on the operator surface. The buttons act on the API
 * endpoints; the truth rows above (source, rotation) always come from
 * the pipeline snapshot. Controls are hidden while the simulated data
 * feed is selected — a control panel on generated numbers would be a lie.
 */
function CameraInputGroup({
  rotationDeg,
  runningIndex,
}: {
  rotationDeg: number;
  runningIndex: number | null;
}) {
  const [sources, setSources] = useState<CameraSourceInfo[] | null>(null);
  const [busy, setBusy] = useState<"" | "switch" | "rotation">("");
  const [error, setError] = useState<string | null>(null);

  const loadSources = useCallback(() => {
    getCameraSources()
      .then((r) => {
        setSources(r.sources);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    loadSources();
  }, [loadSources]);

  async function applySwitch(index: number) {
    if (busy) return;
    setBusy("switch");
    setError(null);
    try {
      const r = await selectCameraSource(index);
      if (!r.applied) setError(r.errors.join(" ") || "Camera switch failed.");
      else loadSources();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function applyRotation(degrees: number) {
    if (busy || degrees === rotationDeg) return;
    setBusy("rotation");
    setError(null);
    try {
      const r = await setCameraRotation(degrees);
      if (!r.applied) setError(r.errors.join(" ") || "Rotation change failed.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  const placeholder = Array.from(
    { length: CAMERA_SOURCES_COUNT },
    (_, i) => ({ index: i, available: false, running: false }),
  );

  return (
    <Group title="Camera input" icon={<Camera size={13} />}>
      <div className="px-md py-xs">
        <div className="text-[12px] text-text-secondary">Camera source</div>
        <div className="flex flex-wrap gap-xs pb-sm pt-xs">
          {(sources ?? placeholder).map((src) => {
            const active = src.running && src.index === runningIndex;
            return (
              <button
                key={src.index}
                type="button"
                className="ctl h-7 px-sm text-[11px]"
                data-active={active}
                disabled={!src.available || busy !== ""}
                onClick={() => applySwitch(src.index)}
                title={
                  src.available
                    ? `Switch to camera ${src.index}`
                    : `Camera ${src.index} is not available`
                }
              >
                {`CAM-${String(src.index + 1).padStart(2, "0")}`}
                {active && <span className="t-meta text-text-tertiary">live</span>}
              </button>
            );
          })}
          {sources === null && (
            <span className="t-meta self-center text-text-tertiary">Detecting cameras…</span>
          )}
        </div>
      </div>
      <div className="px-md pt-xs">
        <div className="text-[12px] text-text-secondary">Image rotation</div>
        <div className="flex gap-xs pb-sm pt-xs">
          {ROTATION_OPTIONS.map((deg) => (
            <button
              key={deg}
              type="button"
              className="ctl h-7 flex-1 justify-center px-0 text-[11px]"
              data-active={rotationDeg === deg}
              disabled={busy !== ""}
              onClick={() => applyRotation(deg)}
            >
              {`${deg}°`}
            </button>
          ))}
        </div>
      </div>
      {busy !== "" && (
        <p className="border-t border-border-primary px-md py-xs text-[11px] text-text-tertiary">
          {busy === "switch" ? "Switching camera…" : "Rotating image…"}
        </p>
      )}
      {error !== null && (
        <p
          className="border-t border-border-primary px-md py-xs text-[11px] font-medium"
          style={{ color: "var(--st-busy)" }}
          role="alert"
        >
          {error}
        </p>
      )}
    </Group>
  );
}


export function DiagnosticsDrawer({
  s,
  open,
  onClose,
  config = null,
  dataSource = "api",
  onDataSource,
  scenario,
  onScenario,
  onShutdown,
}: {
  s: Snapshot;
  open: boolean;
  onClose: () => void;
  config?: ConfigResponse | null;
  dataSource?: "api" | "simulator";
  onDataSource?: (s: "api" | "simulator") => void;
  scenario: Scenario;
  onScenario: (x: Scenario) => void;
  onShutdown: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  // Escape closes, focus moves into the drawer, and focus returns to
  // whatever opened it.
  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    const panel = panelRef.current;
    panel?.querySelector<HTMLElement>("button, [href], input, select, textarea")?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab" || !panel) return;
      const focusable = Array.from(
        panel.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => el.offsetParent !== null);
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      restoreRef.current?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;

  const c = s.camera;
  const cv = s.cv;
  const technicalEvents = s.events.filter((e) => e.technical).slice(0, 8);

  return (
    <div className="fixed inset-0 z-30 flex justify-end" style={{ background: "var(--modal-scrim)" }}>
      {/* Clicking the scrim closes the drawer. */}
      <button
        type="button"
        aria-label="Close diagnostics"
        className="absolute inset-0 cursor-default"
        onClick={onClose}
        tabIndex={-1}
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label="System diagnostics"
        className="relative flex h-full w-full max-w-[440px] flex-col overflow-y-auto border-l border-border-primary bg-surface-bg"
      >
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border-primary bg-surface-bg px-lg py-md">
          <h2 className="t-title text-text-primary">System diagnostics</h2>
          <button
            type="button"
            className="ctl h-7 w-7 justify-center px-0"
            onClick={onClose}
            aria-label="Close diagnostics"
          >
            <X size={14} />
          </button>
        </div>

        <div className="flex flex-col gap-lg px-lg py-lg">
          {/* Source mode */}
          <section>
            <div className="mb-xs flex items-center justify-between">
              <span className="t-label text-text-tertiary">Data mode</span>
              <span
                className="rounded-corner-sm px-sm py-0.5 text-[11px] font-semibold"
                style={{
                  color: dataSource === "api" ? "var(--st-normal)" : "var(--st-busy)",
                  background: `color-mix(in srgb, ${
                    dataSource === "api" ? "var(--st-normal)" : "var(--st-busy)"
                  } 12%, transparent)`,
                }}
              >
                {dataSource === "api" ? "Live camera" : "Simulated data"}
              </span>
            </div>
            <p className="t-body mb-sm text-text-tertiary">
              {dataSource === "api"
                ? "Every value below comes from the detection pipeline running on this machine."
                : "These values are generated by the built-in simulator. They are not measurements."}
            </p>
            <div className="flex gap-xs">
              <button
                type="button"
                className="ctl flex-1 justify-center"
                data-active={dataSource === "api"}
                aria-pressed={dataSource === "api"}
                onClick={() => onDataSource?.("api")}
              >
                Live camera
              </button>
              <button
                type="button"
                className="ctl flex-1 justify-center"
                data-active={dataSource === "simulator"}
                aria-pressed={dataSource === "simulator"}
                onClick={() => onDataSource?.("simulator")}
              >
                Simulator
              </button>
            </div>
          </section>

          <Group title="Camera feed" icon={<Camera size={13} />}>
            <Row label="Connection" state={healthStyle(c.link)} />
            <Row label="Source" value={c.sourceLabel && c.sourceLabel !== "" ? c.sourceLabel : "Not reported"} />
            <Row
              label="Image rotation"
              value={
                c.rotationDeg
                  ? `${c.rotationDeg}° clockwise`
                  : "None (upright)"
              }
            />
            <Row label="Resolution" value={c.resolution} />
            <Row label="Frame rate" value={`${s.stateActive ? c.fps : 0} fps`} />
            <Row label="Latency" value={c.latencyMs === null ? "Not measured" : `${c.latencyMs} ms`} />
            <Row label="Packet loss" value={c.packetLoss === null ? "Not measured" : `${c.packetLoss}%`} />
            <Row label="Last frame" value={lastFrameLabel(s)} />
            <Row label="Uptime" value={fmtDuration(c.uptimeSec)} />
          </Group>

          {/* Setup controls live in diagnostics, off the operator surface.
              Not offered on the simulated feed: controls on generated data
              would be a lie. */}
          {dataSource === "api" && (
            <CameraInputGroup
              rotationDeg={c.rotationDeg ?? 0}
              runningIndex={c.sourceIndex}
            />
          )}

          <Group title="Queue geometry" icon={<Sliders size={13} />}>
            <Row
              label="Queue area"
              state={
                config?.roi_configured
                  ? { color: "var(--st-normal)", label: "Configured", glyph: "circle" }
                  : { color: "var(--st-congested)", label: "Not configured", glyph: "square" }
              }
            />
            {config?.roi_polygon && <Row label="Boundary points" value={`${config.roi_polygon.length}`} />}
            <Row
              label="Counting line"
              state={
                config?.line_configured
                  ? { color: "var(--st-normal)", label: "Configured", glyph: "circle" }
                  : { color: "var(--st-congested)", label: "Not configured", glyph: "square" }
              }
            />
            {config?.xray_side && (
              <Row label="X-ray side" value={config.xray_side === "left" ? "Left" : "Right"} />
            )}
            <Row label="Calibration" value={calibrationLabel(config)} />
            {config?.wait_alert_seconds != null && (
              <Row label="Long wait threshold" value={`${fmtDuration(config.wait_alert_seconds)}`} />
            )}
            <p className="py-xs text-[11px] text-text-tertiary">
              Layout tool: <code className="text-text-secondary">./run.sh --define-roi</code>
            </p>
          </Group>

          <Group title="Detection system" icon={<Activity size={13} />}>
            <Row label="Detection" state={healthStyle(cv.detection)} />
            <Row label="Tracking" state={healthStyle(cv.tracking)} />
            <Row label="Detection model" value={cv.model} />
            <Row label="Tracking method" value={cv.tracker} />
            <Row label="Inference" value={`${cv.inferenceMs.toFixed(1)} ms`} />
            <Row label="Device" value={cv.device || "Not reported"} />
            <Row label="People detected" value={`${cv.detections}`} />
            <Row label="Active tracks" value={`${cv.activeTracks}`} />
          </Group>

          <Group title="Recording" icon={<ShieldCheck size={13} />}>
            <Row label="Status" state={recStyle(s)} />
            <Row
              label="Segment length"
              value={s.recording.sinceSec === null ? "Not available" : fmtDuration(s.recording.sinceSec)}
            />
            <Row
              label="Buffer size"
              value={s.recording.sizeGb === null ? "Not available" : `${s.recording.sizeGb} GB`}
            />
          </Group>

          {technicalEvents.length > 0 && (
            <Group title="Recent events (technical)" icon={<Terminal size={13} />}>
              {technicalEvents.map((e) => (
                <div
                  key={e.id}
                  className="flex items-center justify-between gap-md border-b border-border-primary py-xs last:border-0"
                >
                  <span className="t-meta text-text-tertiary">{fmtClock(e.t)}</span>
                  <span className="t-meta truncate text-text-secondary">{e.technical}</span>
                </div>
              ))}
            </Group>
          )}

          {/* Demo controls live here, never in the operator header. */}
          {dataSource === "simulator" && (
            <section className="border-t border-border-primary pt-lg">
              <div className="mb-sm flex items-center gap-xs">
                <span className="t-label text-text-tertiary">Simulation scenario</span>
              </div>
              <div className="flex flex-col gap-xs">
                {SCENARIOS.map((sc) => {
                  const st = QUEUE_STYLE[sc];
                  const active = scenario === sc;
                  return (
                    <button
                      key={sc}
                      type="button"
                      onClick={() => onScenario(sc)}
                      aria-pressed={active}
                      className={`flex flex-col items-start gap-0.5 rounded-corner-sm border p-sm text-left transition-colors ${
                        active
                          ? "border-control bg-bg-subtle"
                          : "border-border-primary bg-surface-secondary-bg hover:bg-bg-hover"
                      }`}
                    >
                      <span className="flex items-center gap-xs">
                        <StatusMark status={st} size={11} />
                        <span className="text-[12px] font-semibold" style={{ color: st.color }}>
                          {st.label}
                        </span>
                        {active && <span className="text-[11px] text-text-tertiary">Active</span>}
                      </span>
                      <span className="text-[11px] leading-snug text-text-tertiary">{st.description}</span>
                    </button>
                  );
                })}
              </div>
            </section>
          )}

          {/* Shutting the pipeline down is a technician action, so it lives
              here rather than in the operator header. */}
          <section className="border-t border-border-primary pt-lg">
            <div className="mb-xs flex items-center gap-xs">
              <Power size={13} className="text-text-tertiary" aria-hidden />
              <span className="t-label text-text-tertiary">Pipeline control</span>
            </div>
            <p className="t-body mb-sm text-text-tertiary">
              Stops the detection pipeline and the API server on this machine. The console stops updating.
            </p>
            <button
              type="button"
              className="ctl"
              style={{
                color: "var(--st-backlog)",
                borderColor: "color-mix(in srgb, var(--st-backlog) 45%, transparent)",
              }}
              onClick={onShutdown}
            >
              Shut down the pipeline
            </button>
          </section>
        </div>
      </div>
    </div>
  );
}

function Group({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="mb-xs flex items-center gap-xs text-text-tertiary">
        {icon}
        <span className="t-label">{title}</span>
      </div>
      <div className="rounded-corner-md border border-border-primary bg-surface-secondary-bg px-md">
        {children}
      </div>
    </section>
  );
}
