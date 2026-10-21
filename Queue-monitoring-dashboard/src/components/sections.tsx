import { useState, useEffect, useRef, useCallback } from "react";
import {
  Eye,
  EyeOff,
  Camera,
  Clock,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Users,
  PencilLine,
} from "lucide-react";
import type { Snapshot, QueueEvent } from "../lib/telemetry";
import { describeEvent } from "../lib/telemetry";
import type { ConfigResponse } from "../lib/api";
import { getVideoStreamUrl } from "../lib/api";
import MappingEditor from "./MappingEditor";
import {
  Panel,
  StatusMark,
  StatusPill,
  QUEUE_STYLE,
  healthStyle,
  fmtInt,
  fmt1,
  fmtMMSS,
  fmtClock,
  fmtDuration,
  fmtSigned,
  MetricTile,
  Segmented,
  OccupancyMeter,
  OccupancyScale,
} from "../lib/ui";
import { Sparkline, TimeSeriesChart, type Band, type Series } from "./charts";

// ─────────────────────────────────────────────────────────────────────────────
// The operator surface.
//
// Each section owns one question and answers it once:
//   Queue status    -> how many, how bad, which way
//   Key metrics     -> how fast is the flow moving
//   Queue & traffic -> what has the queue been doing
//   Waiting time    -> how long are people being held
//   Live camera     -> what does the checkpoint look like right now
//   Recent activity -> what just happened
//
// Nothing is repeated between sections. The queue count, the capacity and
// the occupancy live in the status panel; the flow rates live in the
// metrics; the waits live in the waiting time panel.
// ─────────────────────────────────────────────────────────────────────────────

export const RANGE_POINTS: Record<string, number> = { "5m": 20, "15m": 60, "30m": 120, "60m": 240 };
export const RANGE_OPTIONS = [
  { id: "5m", label: "5m" },
  { id: "15m", label: "15m" },
  { id: "30m", label: "30m" },
  { id: "60m", label: "60m" },
];

// ── Queue status ───────────────────────────────────────────────────────────
// The dominant element on the screen. It answers "is everything okay?" and
// "how many people are waiting?" in one glance.

export function QueueStatusPanel({ s }: { s: Snapshot }) {
  const active = s.stateActive;
  const st = active ? QUEUE_STYLE[s.state] : QUEUE_STYLE.OFFLINE;
  const critical = active && (s.state === "CONGESTED" || s.state === "BACKLOG");
  const count = active ? s.queue : s.lastKnownQueue;

  const trend = s.trend;
  const trendColor =
    trend === "GROWING"
      ? "var(--st-congested)"
      : trend === "CLEARING"
        ? "var(--st-normal)"
        : "var(--text-secondary)";
  const TrendIcon = trend === "GROWING" ? TrendingUp : trend === "CLEARING" ? TrendingDown : Minus;
  const trendLabel = trend === "GROWING" ? "Naik" : trend === "CLEARING" ? "Turun" : "Stabil";

  // The guidance line comes from the backend's recommendation table, so the
  // wording is configurable and never invented by the dashboard.
  const showAction = active && s.action && s.state !== "NORMAL";

  return (
    <section className="panel relative flex shrink-0 flex-col overflow-hidden">
      {/* One accent stripe carries the condition colour. */}
      <div className="absolute inset-y-0 left-0 w-[3px]" style={{ background: st.color }} aria-hidden />

      <header className="panel-head">
        <div className="flex min-w-0 items-baseline gap-xs">
          <h2 className="t-title">Queue status</h2>
          <span className="truncate text-[11px] text-text-tertiary">Security Checkpoint A</span>
        </div>
        {!active && <StatusPill status={QUEUE_STYLE.OFFLINE} />}
      </header>

      <div className="flex flex-col gap-lg px-lg py-lg">
        <div className="flex flex-wrap items-start justify-between gap-lg">
          <div className="min-w-0">
            <div className="flex items-center gap-sm" style={{ color: st.color }}>
              <StatusMark status={st} size={19} />
              <span className="t-state">{st.label}</span>
            </div>
            <p className="t-body mt-xs max-w-[40ch] text-text-secondary">{st.description}</p>
            {/* The reason only adds something when the condition is not
                normal; otherwise it repeats the sentence above. */}
            {active && s.state !== "NORMAL" && s.reason && s.reason !== st.description && (
              <p className="t-body mt-xs max-w-[46ch] text-text-tertiary">{s.reason}</p>
            )}
            {showAction && (
              <p
                className="t-body mt-sm flex max-w-[52ch] items-start gap-xs rounded-corner-sm px-sm py-xs"
                style={{
                  color: critical ? st.color : "var(--text-secondary)",
                  background: `color-mix(in srgb, ${st.color} 9%, transparent)`,
                  border: `1px solid color-mix(in srgb, ${st.color} 26%, transparent)`,
                }}
              >
                <span className="shrink-0">
                  <StatusMark status={st} size={13} />
                </span>
                {s.action}
              </p>
            )}
          </div>

          <div className="shrink-0 text-right">
            <div className="t-label text-text-tertiary">{active ? "People waiting" : "Last known queue"}</div>
            <div
              className="t-display mt-1"
              style={{ color: active ? "var(--text-primary)" : "var(--text-tertiary)" }}
            >
              {fmtInt(count)}
            </div>
            {active ? (
              <div className="t-meta mt-1 text-text-secondary">
                {s.occupancy.toFixed(0)}% of {s.capacity} places
              </div>
            ) : (
              <div className="t-meta mt-1 text-text-tertiary">Last update {fmtClock(s.lastUpdate)}</div>
            )}
          </div>
        </div>

        <div>
          <div className="mb-xs flex items-baseline justify-between">
            <span className="t-label text-text-tertiary">Occupancy</span>
            <span className="t-meta text-text-secondary">
              {active ? `${s.queue} / ${s.capacity}` : "No signal"}
            </span>
          </div>
          <OccupancyMeter
            occupancy={active ? s.occupancy : 0}
            color={st.color}
            capacity={s.capacity}
            disabled={!active}
          />
          <div className="mt-xs">
            <OccupancyScale />
          </div>
        </div>
      </div>

      {/* Direction, average wait and the worst case, side by side. */}
      <div className="grid shrink-0 grid-cols-3 border-t border-border-primary">
        <StripCell label="Direction">
          {active ? (
            <span className="flex items-center gap-xs text-[17px] font-semibold" style={{ color: trendColor }}>
              <TrendIcon size={15} aria-hidden />
              {trendLabel}
            </span>
          ) : (
            <span className="text-[17px] font-semibold text-text-tertiary">No signal</span>
          )}
        </StripCell>
        <StripCell label="Average wait">
          <span className="t-meta text-[17px] font-semibold text-text-primary">
            {active ? fmtMMSS(s.avgWait) : "--:--"}
          </span>
        </StripCell>
        <StripCell label={critical ? "Congestion" : "Longest wait"}>
          <span
            className="t-meta text-[17px] font-semibold"
            style={{ color: critical ? st.color : "var(--text-primary)" }}
          >
            {!active
              ? "--:--"
              : critical && s.congestionSec > 0
                ? fmtDuration(s.congestionSec)
                : fmtMMSS(s.longestWait)}
          </span>
        </StripCell>
      </div>
    </section>
  );
}

function StripCell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface-secondary-bg px-lg py-sm">
      <div className="t-label text-text-tertiary">{label}</div>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

// ── Key metrics ────────────────────────────────────────────────────────────
// Flow only. The queue size is in the status panel and the waits are in the
// waiting time panel, so neither is repeated here.

export function MetricGrid({ s }: { s: Snapshot }) {
  const h = s.history.slice(-40);
  const active = s.stateActive;
  const net = s.netChange;

  return (
    <Panel title="Key metrics" icon={<Activity size={14} />} className="shrink-0" bodyClassName="p-md">
      <div className="grid grid-cols-2 gap-md">
        <MetricTile
          label="Entry rate"
          value={active ? fmt1(s.entryRate) : "--"}
          unit="/min"
          accent="var(--chart-entry)"
          trend={
            active ? (
              <Sparkline data={h.map((d) => d.entryRate)} color="var(--chart-entry)" height={18} />
            ) : null
          }
        />
        <MetricTile
          label="Exit rate"
          value={active ? fmt1(s.exitRate) : "--"}
          unit="/min"
          accent="var(--chart-exit)"
          trend={
            active ? (
              <Sparkline data={h.map((d) => d.exitRate)} color="var(--chart-exit)" height={18} />
            ) : null
          }
        />
        <MetricTile
          label="Throughput"
          value={active ? fmtInt(s.throughput) : "--"}
          unit="/hr"
          accent="var(--st-normal)"
          footnote={active ? `${fmt1(s.exitRate)} cleared / min` : undefined}
        />
        <MetricTile
          label="Net change"
          value={active ? fmtSigned(net) : "--"}
          unit="/min"
          accent={net > 0 ? "var(--st-congested)" : net < 0 ? "var(--st-normal)" : "var(--text-tertiary)"}
          footnote={active ? (net > 0 ? "Filling" : net < 0 ? "Clearing" : "Balanced") : undefined}
        />
      </div>
    </Panel>
  );
}

// ── Queue & traffic ────────────────────────────────────────────────────────
// The queue length against capacity, and the two flow rates that move it.

export function AnalyticsPanel({
  s,
  range,
  onRange,
}: {
  s: Snapshot;
  range: string;
  onRange: (r: string) => void;
}) {
  const n = RANGE_POINTS[range] ?? 60;
  const data = s.history.slice(-n);
  const active = s.stateActive;

  const capacity = s.capacity || 30;

  const queueSeries: Series[] = [
    {
      id: "queue",
      label: "In queue",
      color: "var(--chart-queue)",
      values: data.map((d) => d.queue),
      fill: true,
    },
  ];

  // The bands are the detector's own occupancy thresholds expressed in
  // people rather than percent, so the chart and the meter agree.
  const bands: Band[] = [
    {
      from: capacity * 0.6,
      to: capacity * 0.8,
      color: "color-mix(in srgb, var(--st-busy) 5%, transparent)",
      label: "Busy",
    },
    {
      from: capacity * 0.8,
      to: capacity,
      color: "color-mix(in srgb, var(--st-congested) 7%, transparent)",
      label: "Congested",
    },
  ];

  const flowSeries: Series[] = [
    { id: "entries", label: "Entries", color: "var(--chart-entry)", values: data.map((d) => d.entryRate) },
    { id: "exits", label: "Exits", color: "var(--chart-exit)", values: data.map((d) => d.exitRate) },
  ];

  return (
    <Panel
      title="Queue & traffic"
      icon={<Activity size={14} />}
      right={<Segmented options={RANGE_OPTIONS} value={range} onChange={onRange} label="Time range" />}
      // Below the two-column breakpoint the panel needs a height of its own:
      // a flex child with no height collapses to its header.
      className="h-[250px] shrink-0 min-h-0 xl:h-auto xl:flex-1"
      bodyClassName="grid min-h-0 grid-cols-1 gap-px bg-border-primary md:grid-cols-2"
    >
      <ChartCell title="Queue length" sub="people waiting">
        {active ? (
          <TimeSeriesChart
            samples={data}
            series={queueSeries}
            bands={bands}
            refs={[{ value: capacity, color: "var(--text-tertiary)", label: `Capacity ${capacity}` }]}
            maxValue={capacity * 1.05}
            formatValue={(v) => String(Math.round(v))}
            ariaLabel={`Queue length over the last ${range}, against a capacity of ${capacity}`}
            emptyMessage="Collecting history"
          />
        ) : (
          <NoSignal />
        )}
      </ChartCell>
      <ChartCell title="Entries vs exits" sub="passengers / min">
        {active ? (
          <TimeSeriesChart
            samples={data}
            series={flowSeries}
            unit="/min"
            formatValue={(v) => v.toFixed(1)}
            ariaLabel={`Entries against exits over the last ${range}`}
            emptyMessage="Collecting history"
          />
        ) : (
          <NoSignal />
        )}
      </ChartCell>
    </Panel>
  );
}

function ChartCell({
  title,
  sub,
  children,
}: {
  title: string;
  sub: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-0 flex-col bg-surface-bg p-md">
      <div className="mb-xs flex items-baseline gap-xs">
        <span className="t-title text-text-secondary">{title}</span>
        <span className="text-[11px] text-text-tertiary">{sub}</span>
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}

// ── Waiting time ────────────────────────────────────────────────────────────
// How long passengers are being held, and how that has moved.

export function WaitingTimePanel({
  s,
  range,
  config,
}: {
  s: Snapshot;
  range: string;
  config: ConfigResponse | null;
}) {
  const n = RANGE_POINTS[range] ?? 60;
  const data = s.history.slice(-n);
  const active = s.stateActive;

  // The threshold is the wait duration the detector itself treats as long
  // enough to contribute to a backlog condition. It comes from the camera
  // configuration, so it is a measured setting and not an invented one.
  const alertMin = config?.wait_alert_seconds ? config.wait_alert_seconds / 60 : null;

  const series: Series[] = [
    {
      id: "wait",
      label: "Average wait",
      color: "var(--chart-wait)",
      values: data.map((d) => d.avgWait),
      fill: true,
    },
  ];

  const refs =
    alertMin !== null
      ? [{ value: alertMin, color: "var(--st-busy)", label: `Long wait ${fmtMMSS(alertMin)}` }]
      : [];

  return (
    <Panel
      title="Waiting time"
      icon={<Clock size={14} />}
      className="h-[210px] shrink-0 min-h-0 xl:h-auto xl:flex-1"
      bodyClassName="flex min-h-0 flex-col"
    >
      <div className="flex shrink-0 items-center justify-between gap-lg px-md pt-md">
        <div className="flex items-baseline gap-lg">
          <Stat label="Average" value={active ? fmtMMSS(s.avgWait) : "--:--"} />
          <Stat label="Longest" value={active ? fmtMMSS(s.longestWait) : "--:--"} />
        </div>
        <span className="text-[11px] text-text-tertiary">Last {range}</span>
      </div>
      <div className="min-h-0 flex-1 p-md">
        {active ? (
          <TimeSeriesChart
            samples={data}
            series={series}
            refs={refs}
            unit="min"
            formatValue={(v) => fmtMMSS(v)}
            ariaLabel={`Average waiting time over the last ${range}`}
            emptyMessage="Collecting history"
          />
        ) : (
          <NoSignal />
        )}
      </div>
    </Panel>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="t-label text-text-tertiary">{label}</div>
      <div className="t-meta mt-0.5 text-[19px] font-semibold text-text-primary">{value}</div>
    </div>
  );
}

function NoSignal() {
  return (
    <div className="flex h-full items-center justify-center text-[12px] text-text-tertiary">No signal</div>
  );
}

// ── Live camera ─────────────────────────────────────────────────────────────
// The feed reads as a fixed airport CCTV camera. The chrome around it is
// drawn in the DOM rather than burned into the video, so it stays crisp.

export function CameraPanel({
  s,
  dataSource = "simulator",
  config = null,
  onConfigChanged,
}: {
  s: Snapshot;
  dataSource?: "simulator" | "api";
  config?: ConfigResponse | null;
  /** Called after the mapping editor persists geometry, to refresh config. */
  onConfigChanged?: () => void;
}) {
  const [show, setShow] = useState({ boxes: true, ids: true, roi: true, line: true });
  const [streamError, setStreamError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const [editing, setEditing] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [panelSize, setPanelSize] = useState({ w: 0, h: 0 });

  const offline = !s.stateActive;
  const degraded = s.camera.link === "DEGRADED";
  const isApi = dataSource === "api";
  // Calibration needs the real stream plus a config to diff against.
  const canEdit = isApi && !offline && config !== null;

  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        const { width, height } = entry.contentRect;
        setPanelSize({ w: width, h: height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!isApi || !imgRef.current) return;
    const img = imgRef.current;
    img.src = getVideoStreamUrl();
    img.onload = () => setStreamError(false);
    img.onerror = () => setStreamError(true);
    return () => {
      img.src = "";
    };
  }, [isApi, reloadKey]);

  const camW = config?.width ?? 0;
  const camH = config?.height ?? 0;

  const oc = objectCoverTransform(panelSize.w, panelSize.h, camW, camH);

  const camToPanel = useCallback(
    (cx: number, cy: number, cw: number = 0, ch: number = 0) => ({
      left: `${((cx * oc.scaleX + oc.offsetX) / panelSize.w) * 100}%`,
      top: `${((cy * oc.scaleY + oc.offsetY) / panelSize.h) * 100}%`,
      width: `${((cw * oc.scaleX) / panelSize.w) * 100}%`,
      height: `${((ch * oc.scaleY) / panelSize.h) * 100}%`,
    }),
    [oc, panelSize],
  );

  const roiConfigured = config?.roi_configured && config.roi_polygon && config.roi_polygon.length >= 3;
  const lineConfigured = config?.line_configured && config.counting_line && config.counting_line.length === 2;

  const roiBBox = roiConfigured
    ? (() => {
        const pts = config!.roi_polygon!;
        let minX = Infinity,
          minY = Infinity,
          maxX = -Infinity,
          maxY = -Infinity;
        for (const [x, y] of pts) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
        return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
      })()
    : null;

  return (
    <Panel
      title="Live camera"
      icon={<Camera size={14} />}
      right={
        <div className="flex items-center gap-md">
          {canEdit && (
            <button
              type="button"
              className="ctl h-6 px-1.5 text-[11px]"
              aria-pressed={editing}
              title="Edit the queue area and IN/OUT counting lines on the live camera"
              onClick={() => setEditing(true)}
            >
              <PencilLine size={13} /> Edit mapping
            </button>
          )}
          {!isApi && (
            <div className="flex items-center gap-xs">
              <OverlayToggle
                on={show.boxes}
                label="People"
                icon={show.boxes ? <Eye size={13} /> : <EyeOff size={13} />}
                onClick={() => setShow((v) => ({ ...v, boxes: !v.boxes }))}
              />
              <OverlayToggle
                on={show.ids}
                label="Ids"
                icon={show.ids ? <Eye size={13} /> : <EyeOff size={13} />}
                onClick={() => setShow((v) => ({ ...v, ids: !v.ids }))}
              />
              <OverlayToggle
                on={show.roi}
                label="Queue area"
                icon={show.roi ? <Eye size={13} /> : <EyeOff size={13} />}
                onClick={() => setShow((v) => ({ ...v, roi: !v.roi }))}
              />
              <OverlayToggle
                on={show.line}
                label="Counting line"
                icon={show.line ? <Eye size={13} /> : <EyeOff size={13} />}
                onClick={() => setShow((v) => ({ ...v, line: !v.line }))}
              />
            </div>
          )}
          <span className="t-meta text-text-tertiary">{offline ? 0 : s.camera.fps} fps</span>
          <StatusPill
            status={
              offline ? QUEUE_STYLE.OFFLINE : degraded ? healthStyle("DEGRADED") : healthStyle("ONLINE")
            }
          />
        </div>
      }
      className="h-[300px] shrink-0 min-h-0 xl:h-auto xl:flex-1"
      bodyClassName="flex flex-col"
    >
      <div ref={panelRef} className="relative min-h-0 flex-1 overflow-hidden bg-black">
        {isApi ? (
          <img
            key={reloadKey}
            ref={imgRef}
            className="absolute inset-0 h-full w-full object-cover"
            alt="Live camera feed of the security screening queue"
            style={{
              filter: offline
                ? "grayscale(1) brightness(0.35)"
                : degraded
                  ? "brightness(0.85) contrast(0.95)"
                  : "none",
              opacity: streamError ? 0.25 : 1,
            }}
          />
        ) : (
          <div
            className="absolute inset-0"
            style={{
              background: "linear-gradient(180deg, #18202c 0%, #101620 55%, #0a0d13 100%)",
              filter: offline
                ? "grayscale(1) brightness(0.35)"
                : degraded
                  ? "brightness(0.85) contrast(0.95)"
                  : "none",
            }}
          />
        )}

        {/* Simulated geometry, so the simulator shows what the detector would. */}
        {!offline && !isApi && (
          <>
            {show.roi && roiBBox && (
              <div className="absolute" style={camToPanel(roiBBox.x, roiBBox.y, roiBBox.w, roiBBox.h)}>
                <div
                  className="h-full w-full border border-dashed"
                  style={{ borderColor: "color-mix(in srgb, var(--st-busy) 60%, transparent)" }}
                />
                <span
                  className="absolute -top-4 left-0 text-[9px] font-semibold tracking-wide"
                  style={{ color: "var(--st-busy)" }}
                >
                  Queue area
                </span>
              </div>
            )}
            {show.line && lineConfigured && (() => {
              const [[lx1, ly1], [lx2, ly2]] = config!.counting_line!;
              return (
                <div
                  className="absolute border-t border-dashed"
                  style={{
                    ...camToPanel(Math.min(lx1, lx2), Math.min(ly1, ly2), Math.abs(lx2 - lx1), 0),
                    borderColor: "color-mix(in srgb, var(--st-busy) 60%, transparent)",
                  }}
                />
              );
            })()}
            {show.boxes &&
              s.tracks.map((t) => (
                <div
                  key={t.id}
                  className="absolute border"
                  style={{
                    left: `${t.x * 100}%`,
                    top: `${t.y * 100}%`,
                    width: `${t.w * 100}%`,
                    height: `${t.h * 100}%`,
                    borderColor: t.inRoi
                      ? "color-mix(in srgb, var(--chart-queue) 70%, transparent)"
                      : "color-mix(in srgb, var(--text-tertiary) 70%, transparent)",
                    borderWidth: 1,
                  }}
                >
                  {show.ids && (
                    <span
                      className="tnum absolute -top-3.5 left-0 text-[9px]"
                      style={{ color: t.inRoi ? "var(--chart-queue)" : "var(--text-tertiary)" }}
                    >
                      {t.id}
                    </span>
                  )}
                </div>
              ))}
          </>
        )}

        {/* CCTV chrome. The checkpoint name is not repeated here: the queue
            status panel above already names it, and a second copy collides
            with the detector's own queue-area caption. */}
        <div className="pointer-events-none absolute left-3 top-2.5 flex flex-col gap-1">
          <span className="text-[11px] font-semibold text-white/85">CAM-01</span>
          {degraded && !offline && (
            <span
              className="w-max rounded-corner-xs px-1 py-0.5 text-[10px] font-semibold"
              style={{
                color: "var(--st-busy)",
                background: "color-mix(in srgb, var(--st-busy) 18%, transparent)",
              }}
            >
              Image quality reduced
            </span>
          )}
        </div>

        {!offline && (
          <div className="pointer-events-none absolute right-2.5 top-2.5 flex items-center gap-1.5">
            <span className="live-dot h-1.5 w-1.5 rounded-full" style={{ background: "var(--st-backlog)" }} />
            <span className="text-[10px] font-semibold tracking-widest text-white/80">LIVE</span>
          </div>
        )}

        <div className="tnum pointer-events-none absolute bottom-1.5 left-2.5 text-[11px] text-white/60">
          {fmtClock(s.frameTs)}
        </div>
        <div className="tnum pointer-events-none absolute bottom-1.5 right-2.5 text-[11px] text-white/45">
          {s.camera.resolution}
        </div>

        {offline && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-sm bg-black/55">
            <StatusMark status={QUEUE_STYLE.OFFLINE} size={22} />
            <span className="t-title text-text-secondary">Signal lost</span>
            <span className="t-meta text-text-tertiary">Last frame {fmtClock(s.lastUpdate)}</span>
          </div>
        )}

        {/* Only meaningful when a real stream is expected: the simulator
            draws its own scene and has no stream to interrupt. */}
        {streamError && !offline && isApi && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-sm bg-black/70">
            <StatusMark status={healthStyle("DEGRADED")} size={22} />
            <span className="t-title text-text-secondary">Camera stream interrupted</span>
            <button type="button" className="ctl" onClick={() => setReloadKey((k) => k + 1)}>
              Reconnect
            </button>
          </div>
        )}

        {/* Interactive calibration overlays everything in this panel;
            monitoring stays untouched while the editor is closed. */}
        {editing && canEdit && (
          <MappingEditor
            config={config}
            onSaved={onConfigChanged}
            onExit={() => setEditing(false)}
          />
        )}
      </div>
    </Panel>
  );
}

function OverlayToggle({
  on,
  label,
  icon,
  onClick,
}: {
  on: boolean;
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="ctl h-6 px-1.5"
      data-active={on}
      aria-pressed={on}
      aria-label={`${label} overlay`}
      title={`${label} overlay`}
      onClick={onClick}
    >
      {icon}
    </button>
  );
}

/**
 * Compute CSS overlays that account for object-cover rendering.
 *
 * object-cover fills the container while maintaining aspect ratio, cropping
 * overflow equally from both sides. The returned {left,top,width,height}%
 * values place an overlay at the correct position within the container so
 * it aligns with the underlying camera image.
 */
function objectCoverTransform(containerW: number, containerH: number, camW: number, camH: number) {
  if (containerW <= 0 || containerH <= 0 || camW <= 0 || camH <= 0) {
    return { offsetX: 0, offsetY: 0, scaleX: 1, scaleY: 1 };
  }
  const imgAspect = camW / camH;
  const panelAspect = containerW / containerH;
  let renderedW: number;
  let renderedH: number;
  if (panelAspect > imgAspect) {
    renderedW = containerH * imgAspect;
    renderedH = containerH;
  } else {
    renderedW = containerW;
    renderedH = containerW / imgAspect;
  }
  const offsetX = (containerW - renderedW) / 2;
  const offsetY = (containerH - renderedH) / 2;
  return {
    offsetX,
    offsetY,
    scaleX: renderedW / camW,
    scaleY: renderedH / camH,
  };
}

// ── Recent activity ─────────────────────────────────────────────────────────
// The operator log. Sentence case, a severity marker, and nothing that names
// an implementation.

const SEVERITY: Record<QueueEvent["severity"], { color: string; label: string }> = {
  info: { color: "var(--text-tertiary)", label: "Info" },
  good: { color: "var(--st-normal)", label: "Good" },
  warn: { color: "var(--st-busy)", label: "Warning" },
  critical: { color: "var(--st-backlog)", label: "Critical" },
};

export function ActivityLog({ s }: { s: Snapshot }) {
  const [filter, setFilter] = useState<"all" | "flow" | "alerts">("all");

  // Newest first. The backend appends events as they occur, so the list
  // arrives in arrival order, which is not the same as time order.
  const events = s.events
    .filter((e) => {
      if (filter === "flow") return e.kind === "entry" || e.kind === "exit";
      if (filter === "alerts") return e.kind === "state" || e.kind === "congestion" || e.kind === "system";
      return true;
    })
    .slice()
    .sort((a, b) => b.t - a.t);

  return (
    <Panel
      title="Recent activity"
      icon={<Users size={14} />}
      right={
        <Segmented
          options={[
            { id: "all", label: "All" },
            { id: "flow", label: "Passenger flow" },
            { id: "alerts", label: "Alerts" },
          ]}
          value={filter}
          onChange={setFilter}
          label="Filter activity"
        />
      }
      className="min-h-0 flex-1"
      bodyClassName="overflow-y-auto"
    >
      {events.length === 0 ? (
        <div className="flex h-full min-h-[64px] items-center justify-center text-[12px] text-text-tertiary">
          Nothing recorded yet
        </div>
      ) : (
        <ul>
          {events.map((e) => {
            const { label, detail } = describeEvent(e);
            const sev = SEVERITY[e.severity];
            return (
              <li
                key={e.id}
                className="flex items-center gap-md border-b border-border-primary px-lg py-xs last:border-0"
              >
                <span className="t-meta w-[58px] shrink-0 text-text-tertiary">{fmtClock(e.t)}</span>
                <span
                  className="h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ background: sev.color }}
                  title={sev.label}
                  aria-label={sev.label}
                />
                <span className="min-w-0 flex-1 truncate text-[13px] text-text-primary">{label}</span>
                {detail && <span className="t-meta shrink-0 text-text-tertiary">{detail}</span>}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
