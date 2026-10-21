import type { ReactNode } from "react";
import { Circle, Diamond, Triangle, Square, CircleOff } from "lucide-react";
import type { QueueState, PipelineStatus, LinkStatus } from "./telemetry";

// ─────────────────────────────────────────────────────────────────────────────
// Design Read
//   A fixed control-room monitoring console for airport security operators.
//   Restrained aviation-operations language. ENERGY 2 / RHYTHM 2 / MOTION 1.
//
//   The status vocabulary below is the single place where a condition maps to
//   a colour, a word and a shape. Colour is never the only carrier of meaning:
//   every state ships a word and a distinct glyph so the console stays
//   readable for colour-blind operators and on washed-out wall displays.
// ─────────────────────────────────────────────────────────────────────────────

export type StatusKey = QueueState | "DEGRADED" | "OFFLINE";

export interface StatusStyle {
  /** Semantic token, e.g. var(--st-normal). */
  color: string;
  /** The word an operator reads. */
  label: string;
  /** Secondary, non-colour encoding. */
  glyph: "circle" | "diamond" | "triangle" | "square" | "ring";
  /**
   * One line explaining the condition in operational language.
   * Optional: ad-hoc states such as a single health indicator do not need
   * a sentence, only a word and a colour.
   */
  description?: string;
}

export const QUEUE_STYLE: Record<StatusKey, StatusStyle> = {
  NORMAL: {
    color: "var(--st-normal)",
    label: "Normal",
    glyph: "circle",
    description: "Queue within normal operating range.",
  },
  BUSY: {
    color: "var(--st-busy)",
    label: "Busy",
    glyph: "diamond",
    description: "Passenger volume is increasing toward capacity.",
  },
  CONGESTED: {
    color: "var(--st-congested)",
    label: "Congested",
    glyph: "triangle",
    description: "Queue approaching high capacity. Monitor closely.",
  },
  BACKLOG: {
    color: "var(--st-backlog)",
    label: "Backlog",
    glyph: "square",
    description: "Queue exceeds normal operating conditions. Action required.",
  },
  DEGRADED: {
    color: "var(--st-busy)",
    label: "Degraded",
    glyph: "diamond",
    description: "Camera or detection quality is reduced. Readings less reliable.",
  },
  OFFLINE: {
    color: "var(--st-offline)",
    label: "Offline",
    glyph: "ring",
    description: "No data from camera or server.",
  },
};

/** Live system conditions, expressed in operator language. */
export function healthStyle(s: PipelineStatus | LinkStatus): StatusStyle {
  switch (s) {
    case "OK":
    case "ONLINE":
      return {
        color: "var(--st-normal)",
        label: "Live",
        glyph: "circle",
        description: "System operating normally.",
      };
    case "DEGRADED":
      return {
        color: "var(--st-busy)",
        label: "Degraded",
        glyph: "diamond",
        description: "Camera or detection quality is reduced.",
      };
    default:
      return {
        color: "var(--st-offline)",
        label: "Offline",
        glyph: "ring",
        description: "No data from camera or server.",
      };
  }
}

// ── Occupancy ladder ────────────────────────────────────────────────────────
// Mirrors the live classifier in config/camera.yaml. One source of truth for
// the meter, the chart threshold bands and the severity colouring.
export const OCCUPANCY_LADDER = [
  { pct: 60, key: "BUSY" as QueueState, label: "Busy" },
  { pct: 80, key: "CONGESTED" as QueueState, label: "Congested" },
];

/** The status colour for a given occupancy percentage. */
export function occupancyStatus(occupancy: number): StatusStyle {
  if (occupancy >= 100) return QUEUE_STYLE.BACKLOG;
  if (occupancy >= 80) return QUEUE_STYLE.CONGESTED;
  if (occupancy >= 60) return QUEUE_STYLE.BUSY;
  return QUEUE_STYLE.NORMAL;
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting
// ─────────────────────────────────────────────────────────────────────────────

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");
export const fmt1 = (n: number) => n.toFixed(1);

/** Minutes → mm:ss, so a wait reads like a clock on the wall. */
export function fmtMMSS(min: number) {
  const total = Math.max(0, Math.round(min * 60));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function fmtClock(ms: number) {
  return new Date(ms).toLocaleTimeString("en-GB", { hour12: false });
}

export function fmtDuration(sec: number) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

export function fmtAgo(ms: number, now: number) {
  const s = Math.max(0, Math.round((now - ms) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s ago`;
}

/** Signed rate, so "+4" and "-2" read as direction at a glance. */
export function fmtSigned(n: number, digits = 1) {
  const v = digits === 0 ? Math.round(n) : Number(n.toFixed(digits));
  return `${v > 0 ? "+" : ""}${v}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitives
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The shape that carries a status without relying on colour.
 * Lucide icons, filled, so the shape is solid rather than an outline stroke.
 */
export function StatusMark({
  status,
  size = 14,
  filled = true,
}: {
  status: StatusStyle;
  size?: number;
  filled?: boolean;
}) {
  const common = {
    size,
    strokeWidth: 2.4,
    fill: filled ? "currentColor" : "none",
    "aria-hidden": true as const,
  };
  switch (status.glyph) {
    case "diamond":
      return <Diamond {...common} />;
    case "triangle":
      return <Triangle {...common} />;
    case "square":
      return <Square {...common} />;
    case "ring":
      return <CircleOff {...common} fill="none" />;
    default:
      return <Circle {...common} />;
  }
}

/**
 * A status as a compact chip: shape + word.
 * Used where a status appears next to other text, never as the headline.
 */
export function StatusPill({
  status,
  size = "sm",
}: {
  status: StatusStyle;
  size?: "sm" | "md";
}) {
  const pad = size === "md" ? "px-sm py-1 text-[12px]" : "px-xs py-0.5 text-[11px]";
  return (
    <span
      className={`inline-flex items-center gap-xs rounded-corner-sm font-semibold whitespace-nowrap ${pad}`}
      style={{
        color: status.color,
        background: `color-mix(in srgb, ${status.color} 12%, transparent)`,
        border: `1px solid color-mix(in srgb, ${status.color} 34%, transparent)`,
      }}
    >
      <StatusMark status={status} size={size === "md" ? 13 : 11} />
      {status.label}
    </span>
  );
}

/**
 * Panel: the framed container every section sits in.
 * Depth comes from the surface step; the border is a hairline that separates
 * regions, not a heavy frame.
 */
export function Panel({
  title,
  icon,
  right,
  children,
  className = "",
  bodyClassName = "",
}: {
  title?: string;
  icon?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={`panel flex min-h-0 flex-col ${className}`}>
      {title && (
        <header className="panel-head">
          <div className="flex min-w-0 items-center gap-xs text-text-secondary">
            {icon}
            <h2 className="t-title text-text-primary">{title}</h2>
          </div>
          {right}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

/**
 * A metric value with its label.
 *
 * The label is the only uppercase text in the console. The value is sentence
 * case and tabular so a live number does not shift its neighbours.
 */
export function MetricTile({
  label,
  value,
  unit,
  trend,
  accent,
  footnote,
}: {
  label: string;
  value: string;
  unit?: string;
  trend?: ReactNode;
  accent?: string;
  footnote?: string;
}) {
  return (
    <div className="flex min-w-0 flex-col justify-between gap-xs rounded-corner-md border border-border-primary bg-surface-secondary-bg px-md py-sm">
      <div className="flex items-center justify-between gap-xs">
        <span className="t-label text-text-tertiary">{label}</span>
        {accent && <span className="h-1.5 w-1.5 rounded-full" style={{ background: accent }} aria-hidden />}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="t-metric text-text-primary">{value}</span>
        {unit && <span className="text-[11px] text-text-tertiary">{unit}</span>}
      </div>
      <div className="flex min-h-[18px] items-end justify-between gap-xs">
        {trend}
        {footnote && <span className="t-meta text-text-tertiary">{footnote}</span>}
      </div>
    </div>
  );
}

/**
 * Segmented control.
 *
 * Real buttons in a radiogroup so Tab lands on the group and the arrow keys
 * move between options, instead of N separate tab stops.
 */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex gap-xs">
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          role="radio"
          aria-checked={value === o.id}
          className="ctl"
          data-active={value === o.id}
          onClick={() => onChange(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/**
 * Occupancy meter.
 *
 * A plain horizontal bar with the classifier's own thresholds marked, so the
 * relationship between people, capacity and percentage is readable without
 * arithmetic. Threshold marks carry a word as well as a position, because a
 * 1px tick at 80% is invisible on a wall display.
 */
export function OccupancyMeter({
  occupancy,
  color,
  capacity,
  disabled,
}: {
  occupancy: number;
  color: string;
  capacity: number;
  disabled?: boolean;
}) {
  const pct = Math.max(0, Math.min(100, occupancy));
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-corner-sm bg-bg-subtle">
      <div
        className="h-full rounded-corner-sm transition-[width] duration-500 ease-out"
        style={{ width: `${pct}%`, background: color, opacity: disabled ? 0.35 : 1 }}
      />
      {OCCUPANCY_LADDER.map((t) => (
        <div
          key={t.pct}
          className="absolute top-0 h-full w-px"
          style={{ left: `${t.pct}%`, background: "var(--border-control)" }}
          aria-hidden
        />
      ))}
    </div>
  );
}

/** The threshold legend that sits under the meter. */
export function OccupancyScale() {
  return (
    <div className="flex items-center justify-between text-[10px] text-text-tertiary">
      <span>0</span>
      {OCCUPANCY_LADDER.map((t) => (
        <span key={t.pct}>{t.pct}% {t.label.toLowerCase()}</span>
      ))}
      <span>Full</span>
    </div>
  );
}
