import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, PencilLine, RotateCcw, X } from "lucide-react";
import type { ConfigResponse, MappingResponse } from "../lib/api";
import { getMapping, putMapping, getVideoStreamUrl, type MappingPayload } from "../lib/api";
import {
  cloneMapping,
  containFit,
  mappingsEqual,
  panelPointToFrame,
  validateMapping,
  type FrameSize,
  type LineId,
  type MappingData,
  type MappingLine,
} from "../lib/mapping";

/**
 * Interactive camera line mapping & calibration.
 *
 * Raw camera picture (annotated=0 stream) + an SVG overlay in normalised
 * frame units positioned with the exact object-cover math from
 * `lib/mapping.ts`, so the geometry sits on the same pixels an operator
 * sees. Editing never leaves the browser until "Save"; the backend is
 * the enforcement mechanism and re-validates everything.
 */

const SVG_VIEW = 1000; // user units across the frame's width; axis 0..1
const STEP = 0.01; // keyboard nudge per arrow press
const STEP_FAST = 0.04; // nudge with Shift held
const MAX_ROI_VERTICES = 12;

const COLOR_ROI = "var(--st-info)";
const COLOR_IN = "var(--st-normal)";
const COLOR_OUT = "var(--st-backlog)";

const LINE_COLOR: Record<LineId, string> = { in: COLOR_IN, out: COLOR_OUT };

interface EditorSaveResult {
  ok: boolean;
  errors?: string[];
}

export default function MappingEditor({
  config,
  onSaved,
  onExit,
}: {
  config: ConfigResponse | null;
  /** Called after a successful save so the parent can refresh. */
  onSaved?: () => void;
  /** Called when the operator closes the editor. */
  onExit: () => void;
}) {
  const [base, setBase] = useState<MappingData | null>(null);
  const [draft, setDraft] = useState<MappingData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<EditorSaveResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [previewSaved, setPreviewSaved] = useState(false); // overlay ghost
  const [frame, setFrame] = useState<FrameSize>({
    width: config?.source_shape_w ?? config?.width ?? 1280,
    height: config?.source_shape_h ?? config?.height ?? 720,
  });
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<{
    kind: "roi" | LineId;
    index: number;
    pointerId: number;
  } | null>(null);
  const [armedAdd, setArmedAdd] = useState(false);
  const [focusHint, setFocusHint] = useState(true);

  // ── load the active mapping once ────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    getMapping()
      .then((resp: MappingResponse) => {
        if (cancelled) return;
        const m = mappingFromWire(resp);
        const source: MappingData =
          m ??
          defaultDraft(config?.source_shape_w, config?.source_shape_h);
        setBase(source);
        setDraft(cloneMapping(source));
      })
      .catch(() => {
        if (!cancelled) setLoadError("The mapping could not be loaded. Check the backend connection and try again.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const panelRef = useRef<HTMLDivElement>(null);
  const [panelSize, setPanelSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setPanelSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fit = useMemo(() => containFit(frame, panelSize), [frame, panelSize]);

  const dirty = useMemo(() => !!(draft && base && !mappingsEqual(draft, base)), [draft, base]);
  const problems = useMemo(
    () => (draft ? validateMapping(draft, frame) : []),
    [draft, frame],
  );
  const roiProblems = problems.filter((p) => p.field === "roi");
  const lineProblems = (["in", "out"] as LineId[]).map((id) => ({
    id,
    items: problems.filter((p) => p.field === id),
  }));

  // ── editing actions ─────────────────────────────────────────────
  const movePoint = useCallback(
    (kind: "roi" | LineId, index: number, u: number, v: number) => {
      setDraft((d) => {
        if (!d) return d;
        const next = cloneMapping(d);
        if (kind === "roi") {
          next.roi[index] = [clamp01(u), clamp01(v)];
        } else {
          next.lines[kind].points[index] = [clamp01(u), clamp01(v)];
        }
        return next;
      });
    },
    [],
  );

  const addRoiVertex = useCallback((u: number, v: number) => {
    setDraft((d) => {
      if (!d || d.roi.length >= MAX_ROI_VERTICES) return d;
      // insert the new vertex into the edge it lands closest to
      const edge = closestEdgeIndex(d.roi, u, v);
      const next = cloneMapping(d);
      next.roi.splice(edge + 1, 0, [clamp01(u), clamp01(v)]);
      return next;
    });
    setArmedAdd(false);
  }, []);

  const removeRoiVertex = useCallback((index: number) => {
    setDraft((d) => {
      if (!d || d.roi.length <= 3) return d;
      const next = cloneMapping(d);
      next.roi.splice(index, 1);
      return next;
    });
  }, []);

  const swapLineEndpoints = useCallback((id: LineId) => {
    setDraft((d) => {
      if (!d) return d;
      const next = cloneMapping(d);
      const line: MappingLine = {
        points: [next.lines[id].points[1], next.lines[id].points[0]],
        direction: next.lines[id].direction,
      };
      next.lines[id] = line;
      return next;
    });
  }, []);

  const resetDraft = useCallback(() => {
    if (base) {
      setDraft(cloneMapping(base));
      setSaveState(null);
    }
  }, [base]);

  const save = useCallback(
    async (closeAfter: boolean) => {
      if (!draft || problems.length > 0 || !dirty) return;
      setSaving(true);
      setSaveState(null);
      try {
        const result = await putMapping(mappingToWire(draft));
        if (result.applied) {
          setBase(cloneMapping(draft));
          setSaveState({ ok: true });
          setPreviewSaved(false);
          onSaved?.();
          if (closeAfter) onExit();
        } else {
          // The backend refused (it is the enforcer): keep the draft and
          // the last known-good mapping active on the backend side.
          setSaveState({ ok: false, errors: result.errors });
        }
      } catch (err) {
        setSaveState({
          ok: false,
          errors: [err instanceof Error ? err.message : "Backend request failed."],
        });
      } finally {
        setSaving(false);
      }
    },
    [draft, problems.length, dirty, onSaved, onExit],
  );

  // ── pointer plumbing (normalised units via the stretched SVG) ───
  const pointerToFrame = useCallback(
    (clientX: number, clientY: number) => {
      const svg = svgRef.current;
      if (!svg) return null;
      const rect = svg.getBoundingClientRect();
      return panelPointToFrame(
        clientX - rect.left,
        clientY - rect.top,
        frame,
        { width: rect.width, height: rect.height },
        "contain",
      );
    },
    [frame],
  );

  useEffect(() => {
    const onMove = (ev: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      const p = pointerToFrame(ev.clientX, ev.clientY);
      if (p) movePoint(drag.kind, drag.index, p.u, p.v);
    };
    const onUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [pointerToFrame, movePoint]);

  const onHandlePointerDown = (kind: "roi" | LineId, index: number) => (ev: React.PointerEvent) => {
    // preventDefault stops text/image drag ghosts but it ALSO cancels the
    // default focus, so the keyboard path would never engage — focus the
    // handle explicitly (same as browsers do for tabindex elements).
    ev.preventDefault();
    const el = ev.currentTarget as Element & { setPointerCapture?: (id: number) => void; focus: () => void };
    el.setPointerCapture?.(ev.pointerId);
    el.focus();
    dragRef.current = { kind, index, pointerId: ev.pointerId };
  };

  const onSvgPointerDown = (ev: React.PointerEvent) => {
    if (!armedAdd || !draft) return;
    const p = pointerToFrame(ev.clientX, ev.clientY);
    if (p) addRoiVertex(p.u, p.v);
  };

  const ghost = previewSaved ? base : null;

  return (
    <div className="absolute inset-0 z-20 flex flex-col bg-black" role="dialog" aria-label="Camera mapping editor">
      {/* raw feed, letterboxed (object-contain) so the editor shows the
          WHOLE frame: handle positions must stay reachable for cropping
          panel shapes — geometry coordinates cover the full frame. */}
      <div ref={panelRef} className="relative min-h-0 flex-1 overflow-hidden">
        <img
          className="absolute inset-0 m-auto max-h-full max-w-full object-contain"
          src={getVideoStreamUrl(false)}
          alt="Raw camera feed for mapping calibration"
        />
        <svg
          ref={svgRef}
          className={`absolute ${armedAdd ? "cursor-crosshair" : ""}`}
          viewBox={`0 0 ${SVG_VIEW} ${SVG_VIEW}`}
          preserveAspectRatio="none"
          style={{
            left: fit.offsetX,
            top: fit.offsetY,
            width: fit.renderW,
            height: fit.renderH,
          }}
          onPointerDown={onSvgPointerDown}
        >
          <defs>
            <marker id="map-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
            </marker>
          </defs>

          {/* previously saved mapping, as the preview ghost */}
          {ghost && <Geometry mapping={ghost} viewBox={SVG_VIEW} muted />}

          {draft && (
            <Geometry
              mapping={draft}
              viewBox={SVG_VIEW}
              onVertexPointerDown={onHandlePointerDown}
              onVertexKeyDown={handleVertexKey(movePoint, removeRoiVertex, (id) => swapLineEndpoints(id))}
              onVertexDoubleClick={(kind, index) => kind === "roi" && removeRoiVertex(index)}
              problems={problems}
            />
          )}
        </svg>

        {/* ── header chrome ─────────────────────────────── */}
        <div className="pointer-events-none absolute left-3 top-2.5 flex items-center gap-2">
          <span className="rounded-corner-xs bg-black/70 px-1.5 py-0.5 text-[10px] font-semibold tracking-widest text-white/85">
            EDIT MAPPING
          </span>
          {dirty && (
            <span className="rounded-corner-xs bg-black/70 px-1.5 py-0.5 text-[10px] font-semibold" style={{ color: "var(--st-busy)" }}>
              Unsaved changes
            </span>
          )}
          {previewSaved && (
            <span className="rounded-corner-xs bg-black/70 px-1.5 py-0.5 text-[10px] text-white/70">
              Dashed = saved geometry
            </span>
          )}
        </div>
      </div>

      {/* ── legend + validation + actions ─────────────────────── */}
      <div className="shrink-0 border-t border-white/10 bg-[#0b0f16] px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
          <Legend swatch={COLOR_ROI} label="Queue area ROI" />
          <Legend swatch={COLOR_IN} label="IN line (entry)" />
          <Legend swatch={COLOR_OUT} label="OUT line (exit)" />
          <span className="text-text-tertiary">
            Drag handles to move · click an edge in "Add point" mode to insert · double-click or Delete removes (min 3) · arrows nudge, Shift = larger
          </span>
        </div>

        {loadError ? (
          <p className="mt-1.5 text-[12px]" style={{ color: "var(--st-backlog)" }}>{loadError}</p>
        ) : draft && draft.roi.length >= 3 ? null : (
          <p className="mt-1.5 text-[12px]" style={{ color: "var(--st-backlog)" }}>
            The queue area needs at least 3 points. Click the frame in "Add point" mode (or use the frame menu) to draw it.
          </p>
        )}
        {saveState && !saveState.ok && (
          <ul className="mt-1.5 list-inside list-disc text-[12px]" style={{ color: "var(--st-backlog)" }}>
            {saveState.errors?.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        )}
        {saveState?.ok && (
          <p className="mt-1.5 text-[12px]" style={{ color: "var(--st-normal)" }}>
            Mapping applied. The detector switched to it on the next frame.
          </p>
        )}
        {!saveState?.ok && roiProblems.length > 0 && (
          <ul className="mt-1.5 list-inside list-disc text-[12px]" style={{ color: "var(--st-backlog)" }}>
            {roiProblems.map((p, i) => <li key={i}>{p.message}</li>)}
          </ul>
        )}
        {!saveState?.ok &&
          lineProblems.map(({ id, items }) =>
            items.length > 0 ? (
              <ul key={id} className="mt-1.5 list-inside list-disc text-[12px]" style={{ color: "var(--st-backlog)" }}>
                {items.map((p, i) => <li key={i}>{p.message}</li>)}
              </ul>
            ) : null,
          )}

        <div className="mt-2 flex flex-wrap items-center gap-xs">
          <button
            type="button"
            className="ctl h-7 px-2"
            data-active={armedAdd}
            aria-pressed={armedAdd}
            disabled={!!draft && draft.roi.length >= MAX_ROI_VERTICES}
            onClick={() => setArmedAdd((v) => !v)}
          >
            <PencilLine size={13} /> Add point{armedAdd ? " (click frame)" : ""}
          </button>
          <button
            type="button"
            className="ctl h-7 px-2"
            aria-pressed={previewSaved}
            onClick={() => setPreviewSaved((v) => !v)}
          >
            Preview saved
          </button>
          <div className="grow" />
          <button type="button" className="ctl h-7 px-2" onClick={resetDraft} disabled={!dirty}>
            <RotateCcw size={13} /> Revert
          </button>
          <button type="button" className="ctl h-7 px-2" onClick={onExit} title="Close without saving. Unsaved changes are discarded.">
            <X size={13} /> Cancel
          </button>
          <button
            type="button"
            className="ctl h-7 px-2.5"
            disabled={saving || !dirty || problems.length > 0}
            onClick={() => save(false)}
            title={problems.length > 0 ? "Fix the highlighted problems to enable saving." : "Persist this mapping to the backend."}
          >
            <Check size={13} /> Save
          </button>
          <button
            type="button"
            className="ctl h-7 px-2.5"
            data-active={dirty && problems.length === 0}
            disabled={saving || !dirty || problems.length > 0}
            onClick={() => save(true)}
            title="Persist this mapping and return to the monitoring view."
          >
            Apply
          </button>
        </div>
        {focusHint && (
          <p className="mt-1.5 text-[10px] text-text-tertiary" tabIndex={-1}>
            Handles are focusable: Tab moves between points, arrow keys nudge, Shift+arrow nudges more, Delete removes ROI vertices.
          </p>
        )}
      </div>
    </div>
  );
}

// ── geometry rendering ───────────────────────────────────────────────────

function Geometry({
  mapping,
  viewBox,
  muted = false,
  onVertexPointerDown,
  onVertexKeyDown,
  onVertexDoubleClick,
  problems = [],
}: {
  mapping: MappingData;
  viewBox: number;
  muted?: boolean;
  onVertexPointerDown?: (kind: "roi" | LineId, index: number) => (ev: React.PointerEvent) => void;
  onVertexKeyDown?: (ev: React.KeyboardEvent) => void;
  onVertexDoubleClick?: (kind: "roi" | LineId, index: number) => void;
  problems?: Array<{ field: string; message: string }>;
}) {
  const px = (u: number) => u * viewBox;
  const py = (v: number) => v * viewBox;
  const s = 3; // half stroke width in user units (vector-normalised)
  const roiInvalid = problems.some((p) => p.field === "roi");
  const dash = muted ? "6 6" : undefined;
  const opacity = muted ? 0.45 : 1;

  const polygon = mapping.roi.map(([u, v]) => `${px(u)},${py(v)}`).join(" ");

  return (
    <g onKeyDown={onVertexKeyDown}>
      {/* queue ROI polygon */}
      <polygon
        points={polygon}
        fill={COLOR_ROI}
        fillOpacity={muted ? 0.06 : 0.14}
        stroke={COLOR_ROI}
        strokeWidth={s}
        strokeDasharray={dash}
        vectorEffect="non-scaling-stroke"
        opacity={opacity}
      />
      {mapping.lines &&
        (Object.keys(mapping.lines) as LineId[]).map((id) => {
          const line = mapping.lines[id];
          const [a, b] = line.points;
          const color = LINE_COLOR[id];
          const invalid = problems.some((p) => p.field === id);
          // forward direction arrow: the positive half-plane of a→b is to
          // the RIGHT of travel in screen coordinates (y down), matching
          // the backend's `_line_side` sign convention.
          const dx = px(b[0]) - px(a[0]);
          const dy = py(b[1]) - py(a[1]);
          const len = Math.hypot(dx, dy) || 1;
          const ax = -dy / len;
          const ay = dx / len;
          const midX = (px(a[0]) + px(b[0])) / 2;
          const midY = (py(a[1]) + py(b[1])) / 2;
          const arrowLen = 60;
          return (
            <g key={id} opacity={opacity}>
              {/* soft band on the forward side, so "in" and "out" keep their */}
              {/* identity even when the values coincide between lines */}
              <line
                x1={px(a[0])} y1={py(a[1])} x2={px(b[0])} y2={py(b[1])}
                stroke={invalid ? "var(--st-backlog)" : color}
                strokeWidth={s}
                strokeDasharray={dash}
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1={midX} y1={midY}
                x2={midX + ax * arrowLen} y2={midY + ay * arrowLen}
                stroke={invalid ? "var(--st-backlog)" : color}
                strokeWidth={s}
                markerEnd="url(#map-arrow)"
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={midX + ax * arrowLen * 2.2}
                y={midY + ay * arrowLen * 2.2}
                fill={color}
                fontSize={34}
                fontWeight={700}
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {id.toUpperCase()}
              </text>
              {line.points.map(([u, v], i) => (
                <circle
                  key={i}
                  cx={px(u)} cy={py(v)} r={9}
                  fill={color}
                  stroke="white"
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                  tabIndex={muted ? undefined : 0}
                  role="button"
                  data-kind={id}
                  data-index={i}
                  data-u={u}
                  data-v={v}
                  aria-label={`${id.toUpperCase()} line point ${i + 1}, x ${Math.round(u * 100)} percent, y ${Math.round(v * 100)} percent. Arrow keys nudge, Enter swaps the direction`}
                  onPointerDown={onVertexPointerDown?.(id, i)}
                  onDoubleClick={muted ? undefined : (ev) => onVertexDoubleClick?.(id, i)}
                  style={{ cursor: muted ? "default" : "grab" }}
                />
              ))}
            </g>
          );
        })}
      {mapping.roi.map(([u, v], i) => (
        <circle
          key={i}
          cx={px(u)} cy={py(v)} r={9}
          fill={roiInvalid ? "var(--st-busy)" : COLOR_ROI}
          stroke="white"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
          tabIndex={muted ? undefined : 0}
          role="button"
          data-kind="roi"
          data-index={i}
          data-u={u}
          data-v={v}
          aria-label={`Queue area point ${i + 1}, x ${Math.round(u * 100)} percent, y ${Math.round(v * 100)} percent. Arrow keys nudge, Delete removes (minimum 3 points)`}
          onPointerDown={onVertexPointerDown?.("roi", i)}
          onDoubleClick={muted ? undefined : (ev) => onVertexDoubleClick?.("roi", i)}
          style={{ cursor: muted ? "default" : "grab" }}
        />
      ))}
    </g>
  );
}

function Legend({ swatch, label }: { swatch: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-text-secondary">
      <span className="inline-block h-2 w-4 rounded-corner-xs" style={{ background: swatch }} />
      {label}
    </span>
  );
}

// ── key/interaction plumbing ─────────────────────────────────────────────

/** Keys on a focused handle: Tab-navigable, arrows nudge, Delete removes.
 *  The handler reads the focused element's `data-*` attributes, so one
 *  handler services every handle. */
function handleVertexKey(
  movePoint: (kind: "roi" | LineId, index: number, u: number, v: number) => void,
  removeRoiVertex: (index: number) => void,
  swapLine: (id: LineId) => void,
) {
  return (ev: React.KeyboardEvent) => {
    const el = ev.target as HTMLElement & { dataset: Record<string, string | undefined> };
    const kind = el.dataset.kind as "roi" | LineId | undefined;
    const index = el.dataset.index ? Number(el.dataset.index) : -1;
    if (!kind || index < 0) return;
    const step = ev.shiftKey ? STEP_FAST : STEP;
    const directions: Record<string, [number, number] | null> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const d = directions[ev.key];
    if (d) {
      ev.preventDefault();
      movePoint(kind, index, Number(el.dataset.u) + d[0], Number(el.dataset.v) + d[1]);
      return;
    }
    if ((ev.key === "Delete" || ev.key === "Backspace") && kind === "roi") {
      ev.preventDefault();
      removeRoiVertex(index);
      return;
    }
    if (ev.key === "Enter" && kind !== "roi") {
      ev.preventDefault();
      swapLine(kind);
    }
  };
}

// ── wire conversion + defaults ───────────────────────────────────────────

/** Backend wire mapping → editor `MappingData` (normalised, in/out keys). */
function mappingFromWire(resp: MappingResponse): MappingData | null {
  const m = resp.mapping;
  if (!m || !Array.isArray(m.roi) || m.roi.length < 3) return null;
  const lines: Partial<Record<LineId, MappingLine>> = {};
  for (const raw of m.lines ?? []) {
    if ((raw.id === "in" || raw.id === "out") && Array.isArray(raw.points) && raw.points.length === 2) {
      lines[raw.id] = {
        points: [[raw.points[0][0], raw.points[0][1]], [raw.points[1][0], raw.points[1][1]]],
        direction: raw.direction,
      };
    }
  }
  if (!lines.in || !lines.out) return null;
  return {
    roi: m.roi.map(([u, v]) => [u, v] as [number, number]),
    lines: { in: lines.in, out: lines.out },
    referencePoint:
      resp.reference_point === "centroid"
        ? "centroid"
        : m.reference_point === "centroid"
          ? "centroid"
          : "bottom_center",
    sourceShape: m.source_shape ?? undefined,
  };
}

function mappingToWire(m: MappingData, configSourceShape?: { width: number; height: number } | null): MappingPayload {
  return {
    schema_version: 1,
    roi: m.roi.map(([u, v]) => [u, v]),
    lines: (["in", "out"] as LineId[]).map((id) => ({
      id,
      points: m.lines[id].points.map(([u, v]) => [u, v]),
      direction: m.lines[id].direction,
    })),
    reference_point: m.referencePoint,
    ...(m.sourceShape ?? configSourceShape
      ? { source_shape: m.sourceShape ?? configSourceShape ?? undefined }
      : {}),
  };
}

/**
 * A sane starting point for a camera with no mapping yet: a centred ROI
 * with the IN line on its left edge and the OUT line on its right edge,
 * so the first save is always an improvement over nothing but still
 * requires an explicit operator adjustment.
 */
function defaultDraft(width?: number, height?: number): MappingData {
  return {
    roi: [
      [0.3, 0.25],
      [0.7, 0.25],
      [0.7, 0.75],
      [0.3, 0.75],
    ],
    lines: {
      in: { points: [[0.3, 0.9], [0.3, 0.6]], direction: "IN" },
      out: { points: [[0.7, 0.6], [0.7, 0.9]], direction: "OUT" },
    },
    referencePoint: "bottom_center",
    sourceShape:
      width && height ? { width, height } : undefined,
  };
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/** Index of the polygon edge (p[i] → p[i+1]) the point lands closest to. */
function closestEdgeIndex(poly: Array<[number, number]>, u: number, v: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const d = distToSegment(u, v, a, b);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

function distToSegment(
  u: number,
  v: number,
  a: [number, number],
  b: [number, number],
): number {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const apx = u - a[0];
  const apy = v - a[1];
  const lenSq = abx * abx + aby * aby;
  const t = lenSq === 0 ? 0 : Math.min(1, Math.max(0, (apx * abx + apy * aby) / lenSq));
  const cx = a[0] + t * abx;
  const cy = a[1] + t * aby;
  return Math.hypot(u - cx, v - cy);
}
