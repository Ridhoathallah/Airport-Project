import { useCallback, useEffect, useRef, useState } from "react";
import type { Sample } from "../lib/telemetry";
import { fmtClock } from "../lib/ui";

// ─────────────────────────────────────────────────────────────────────────────
// Time-series charts.
//
// The previous implementation drew into a fixed 520x150 viewBox and let the
// SVG scale with `preserveAspectRatio`, which meant the axis labels were
// sized for a 520px box, the strokes were scaled unevenly, and the x-axis
// captions escaped the panel. Every chart here is measured instead: the SVG
// is drawn at the container's real pixel size, so grid lines are 1px, labels
// are legible, and nothing can overflow.
// ─────────────────────────────────────────────────────────────────────────────

export interface Series {
  id: string;
  label: string;
  color: string;
  /** One value per sample, in the same order as the samples. */
  values: number[];
  /** Fill the area under the line. */
  fill?: boolean;
  /** Reference line rather than a data series. */
  dashed?: boolean;
}

export interface Band {
  from: number;
  to: number;
  color: string;
  label?: string;
}

export interface RefLine {
  value: number;
  color: string;
  label: string;
}

const AXIS_FONT = 10;

/**
 * A readable axis: a round top and a round step.
 *
 * Dividing the maximum by a fixed number produced ticks like 13, 25, 38, 50
 * and 01:15, 02:30, 03:45, which an operator has to decode. This picks a
 * step from the 1 / 2 / 2.5 / 5 / 10 family and rounds the axis top up to a
 * multiple of it, so every label is a number someone would say out loud.
 */
function niceScale(maxRaw: number, targetTicks = 4) {
  if (maxRaw <= 0) return { max: 1, step: 1 };
  const rawStep = maxRaw / targetTicks;
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const n = rawStep / pow;
  const mult = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
  const step = mult * pow;
  const max = Math.ceil(maxRaw / step - 1e-9) * step;
  return { max, step };
}

/** Measure a container so the chart can be drawn at its real size. */
function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setSize({ w: r.width, h: r.height });
    });
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  return [ref, size] as const;
}

/** The value axis ticks for a 0..max range. */
function ticks(max: number, step: number) {
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) {
    out.push(Math.round(v * 1000) / 1000);
  }
  return out;
}

export interface TimeSeriesChartProps {
  samples: Sample[];
  series: Series[];
  bands?: Band[];
  refs?: RefLine[];
  /** Suffix for the value axis, e.g. "min". */
  unit?: string;
  /** Format one axis value. */
  formatValue?: (v: number) => string;
  /** Label for the whole chart, used as the accessible name. */
  ariaLabel: string;
  emptyMessage?: string;
  /** Fixed axis top. Use when a reference line must stay visible even when
   *  the data is far below it, e.g. the queue capacity against a short queue. */
  maxValue?: number;
}

/**
 * A multi-series time chart with hover and keyboard inspection.
 *
 * Keyboard: the chart is a single tab stop. Arrow keys move the inspection
 * point, Home/End jump to the ends, and the values are announced through an
 * aria-live region, so the data is reachable without a mouse.
 */
export function TimeSeriesChart({
  samples,
  series,
  bands = [],
  refs = [],
  unit,
  formatValue,
  ariaLabel,
  emptyMessage = "No data",
  maxValue,
}: TimeSeriesChartProps) {
  const [wrapRef, size] = useSize<HTMLDivElement>();
  const [cursor, setCursor] = useState<number | null>(null);
  const n = samples.length;

  const fmt = formatValue ?? ((v: number) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)));

  const pad = { l: 34, r: 10, t: 10, b: 18 };
  const iw = Math.max(0, size.w - pad.l - pad.r);
  const ih = Math.max(0, size.h - pad.t - pad.b);

  const scale = niceScale(
    Math.max(
      1,
      maxValue ?? 0,
      ...series.flatMap((s) => s.values),
      ...refs.map((r) => r.value),
      ...bands.map((b) => b.to),
    ),
  );
  const max = scale.max;

  const x = useCallback(
    (i: number) => pad.l + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw),
    [iw, n, pad.l],
  );
  const y = useCallback((v: number) => pad.t + ih - (Math.max(0, Math.min(v, max)) / max) * ih, [ih, max, pad.t]);

  const onMove = useCallback(
    (e: React.MouseEvent) => {
      const el = e.currentTarget;
      const rect = el.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const i = Math.round(((px - pad.l) / (iw || 1)) * (n - 1));
      setCursor(Math.max(0, Math.min(n - 1, i)));
    },
    [iw, n, pad.l],
  );

  const onKey = useCallback(
    (e: React.KeyboardEvent) => {
      if (n === 0) return;
      const step = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setCursor((c) => Math.min(n - 1, (c ?? n - 1) + step));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCursor((c) => Math.max(0, (c ?? n - 1) - step));
      } else if (e.key === "Home") {
        e.preventDefault();
        setCursor(0);
      } else if (e.key === "End") {
        e.preventDefault();
        setCursor(n - 1);
      } else if (e.key === "Escape") {
        setCursor(null);
      }
    },
    [n],
  );

  if (n === 0) {
    return (
      <div className="flex h-full items-center justify-center text-[12px] text-text-tertiary">{emptyMessage}</div>
    );
  }

  const active = cursor != null ? Math.min(cursor, n - 1) : n - 1;
  const sample = samples[active];

  const gridTicks = ticks(max, scale.step);

  return (
    <div ref={wrapRef} className="relative h-full w-full">
      {size.w > 0 && size.h > 0 && (
        <svg
          width={size.w}
          height={size.h}
          role="img"
          tabIndex={0}
          aria-label={ariaLabel}
          onMouseMove={onMove}
          onMouseLeave={() => setCursor(null)}
          onKeyDown={onKey}
          onFocus={() => setCursor((c) => c ?? n - 1)}
          onBlur={() => setCursor(null)}
          className="block"
        >
          {/* Threshold bands sit behind everything: they are context, not data. */}
          {bands.map((b) => (
            <rect
              key={`${b.from}-${b.to}`}
              x={pad.l}
              y={y(b.to)}
              width={iw}
              height={Math.max(0, y(b.from) - y(b.to))}
              fill={b.color}
            />
          ))}

          {/* Grid */}
          {gridTicks.map((t) => (
            <g key={t}>
              <line
                x1={pad.l}
                x2={pad.l + iw}
                y1={y(t)}
                y2={y(t)}
                stroke="var(--border-primary)"
                strokeWidth={1}
                shapeRendering="crispEdges"
              />
              <text
                x={pad.l - 6}
                y={y(t) + 3}
                fontSize={AXIS_FONT}
                fill="var(--text-tertiary)"
                textAnchor="end"
                className="tnum"
              >
                {fmt(t)}
              </text>
            </g>
          ))}

          {/* Reference lines */}
          {refs.map((r) => (
            <g key={r.label}>
              <line
                x1={pad.l}
                x2={pad.l + iw}
                y1={y(r.value)}
                y2={y(r.value)}
                stroke={r.color}
                strokeWidth={1}
                strokeDasharray="4 3"
              />
              <text
                x={pad.l + iw - 2}
                y={y(r.value) - 5}
                fontSize={AXIS_FONT}
                fill={r.color}
                textAnchor="end"
                className="tnum"
              >
                {r.label}
              </text>
            </g>
          ))}

          {/* Series */}
          {series.map((s) => {
            const pts = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
            const area =
              s.fill && pts.length
                ? `M${pad.l},${y(0).toFixed(1)} L${pts.join(" L")} L${(pad.l + iw).toFixed(1)},${y(0).toFixed(1)} Z`
                : "";
            return (
              <g key={s.id}>
                {area && <path d={area} fill={s.color} opacity={0.12} />}
                <polyline
                  points={pts.join(" ")}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={s.dashed ? 1.25 : 1.75}
                  strokeDasharray={s.dashed ? "4 3" : undefined}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              </g>
            );
          })}

          {/* Inspection point */}
          {cursor != null && (
            <g>
              <line
                x1={x(active)}
                x2={x(active)}
                y1={pad.t}
                y2={pad.t + ih}
                stroke="var(--text-tertiary)"
                strokeWidth={1}
              />
              {series.map((s) => (
                <circle
                  key={s.id}
                  cx={x(active)}
                  cy={y(s.values[active] ?? 0)}
                  r={2.75}
                  fill={s.color}
                  stroke="var(--surface-bg)"
                  strokeWidth={1.5}
                />
              ))}
            </g>
          )}

          {/* Time axis: only the two ends, so the axis never crowds the plot. */}
          <text x={pad.l} y={size.h - 4} fontSize={AXIS_FONT} fill="var(--text-tertiary)" className="tnum">
            {fmtClock(samples[0].t)}
          </text>
          <text
            x={pad.l + iw}
            y={size.h - 4}
            fontSize={AXIS_FONT}
            fill="var(--text-tertiary)"
            textAnchor="end"
            className="tnum"
          >
            {fmtClock(samples[n - 1].t)}
          </text>
        </svg>
      )}

      {cursor != null && (
        <ChartTip
          sample={sample}
          series={series}
          index={active}
          unit={unit}
          fmt={fmt}
          x={x(active)}
          width={size.w}
        />
      )}

      {/* Announced to assistive technology when the operator inspects with keys. */}
      <p className="sr-only" aria-live="polite">
        {cursor != null
          ? `${fmtClock(sample.t)}: ${series
              .map((s) => `${s.label} ${fmt(s.values[active] ?? 0)}${unit ? ` ${unit}` : ""}`)
              .join(", ")}`
          : ""}
      </p>
    </div>
  );
}

function ChartTip({
  sample,
  series,
  index,
  unit,
  fmt,
  x,
  width,
}: {
  sample: Sample;
  series: Series[];
  index: number;
  unit?: string;
  fmt: (v: number) => string;
  x: number;
  width: number;
}) {
  // Keep the tooltip inside the chart instead of letting it hang off the edge.
  const flip = x > width * 0.62;
  return (
    <div
      className="pointer-events-none absolute top-1 z-10 rounded-corner-sm border border-border-primary bg-surface-dark px-sm py-xs shadow-lg"
      style={flip ? { right: Math.max(4, width - x + 8) } : { left: Math.min(Math.max(4, x + 8), Math.max(4, width - 150)) }}
    >
      <div className="t-meta mb-0.5 text-text-tertiary">{fmtClock(sample.t)}</div>
      {series.map((s) => (
        <div key={s.id} className="flex items-center justify-between gap-md">
          <span className="flex items-center gap-xs text-[11px] text-text-secondary">
            <span className="h-[2px] w-3 rounded" style={{ background: s.color }} aria-hidden />
            {s.label}
          </span>
          <span className="tnum text-[11px] font-semibold text-text-primary">
            {fmt(s.values[index] ?? 0)}
            {unit ? ` ${unit}` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * Sparkline: an axis-free micro-trend for a metric tile.
 * No text, so a stretched viewBox is safe here.
 */
export function Sparkline({
  data,
  color = "var(--st-info)",
  height = 18,
}: {
  data: number[];
  color?: string;
  height?: number;
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const w = 100;
  const h = height;
  if (data.length < 2) return <div ref={ref} style={{ height }} className="w-full" />;

  const max = Math.max(...data);
  const min = Math.min(...data);
  const span = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - 1.5 - ((v - min) / span) * (h - 3);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const last = pts[pts.length - 1].split(",");

  return (
    <div ref={ref} className="w-full" style={{ height }}>
      {size.w > 0 && (
        <svg
          width={size.w}
          height={h}
          viewBox={`0 0 ${w} ${h}`}
          preserveAspectRatio="none"
          aria-hidden
          className="block"
        >
          <polyline
            points={pts.join(" ")}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
          />
          <circle cx={last[0]} cy={last[1]} r={2} fill={color} vectorEffect="non-scaling-stroke" />
        </svg>
      )}
    </div>
  );
}
