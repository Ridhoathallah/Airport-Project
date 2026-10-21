/**
 * Camera mapping geometry — the single client-side source of truth.
 *
 * Everything here is PURE: no DOM, no fetch. The mapping editor and the
 * App both import from this module, and `mapping.test.ts` unit-tests the
 * exact functions that run in production (object-cover scaling,
 * letterboxing and pointer→frame mapping), including exact numbers.
 *
 * All mapping coordinates are NORMALISED to the frame: x, y ∈ [0, 1].
 * Conversions to panel pixels happen only at render/pointer time.
 */

export type LineId = "in" | "out";
export type LineDirection = "IN" | "OUT";

export interface MappingLine {
  /** Exactly two endpoints, normalised. */
  points: [[number, number], [number, number]];
  direction: LineDirection;
}

export interface MappingData {
  /** Queue ROI polygon vertices, normalised, in draw order. */
  roi: Array<[number, number]>;
  /** Separate IN and OUT counting lines. */
  lines: Record<LineId, MappingLine>;
  referencePoint: "bottom_center" | "centroid";
  /** Frame the coordinates were drawn against (diagnostics). */
  sourceShape?: { width: number; height: number };
}

export interface FrameSize {
  width: number;
  height: number;
}

export interface PanelSize {
  width: number;
  height: number;
}

/** How a frame of `frame` px is rendered inside a `box` px container. */
export interface CoverFit {
  /** Rendered size of the whole frame inside the panel. */
  renderW: number;
  renderH: number;
  /** Offset of the rendered frame inside the panel. Negative with cover
   *  (the overflow is cropped by the container); positive with contain
   *  (the letterbox bars the container shows around the frame). */
  offsetX: number;
  offsetY: number;
  /** frame px → panel px scale factors. */
  scale: number;
}

export type FitMode = "cover" | "contain";

/**
 * `object-fit: contain`: the frame is fully visible, letterboxed inside
 * the container. The mapping editor uses this so every vertex of the
 * queue ROI stays reachable no matter how wide the panel is — geometry
 * coordinates are normalised to the FULL frame, which cover would crop.
 */
export function containFit(frame: FrameSize, box: PanelSize): CoverFit {
  if (
    box.width <= 0 ||
    box.height <= 0 ||
    frame.width <= 0 ||
    frame.height <= 0
  ) {
    return { renderW: 0, renderH: 0, offsetX: 0, offsetY: 0, scale: 0 };
  }
  const scale = Math.min(box.width / frame.width, box.height / frame.height);
  const renderW = frame.width * scale;
  const renderH = frame.height * scale;
  return {
    renderW,
    renderH,
    offsetX: (box.width - renderW) / 2,
    offsetY: (box.height - renderH) / 2,
    scale,
  };
}

/**
 * True `object-fit: cover` for a frame inside a container: the frame is
 * scaled so BOTH dimensions fill the container while maintaining aspect
 * ratio; overflow is cropped equally from both sides. This matches the
 * CSS the camera <img> uses, so overlays land on the same pixels the
 * operator sees. (The previous helper computed a contain/letterbox fit
 * while the image rendered with cover — this one is the cover math.)
 */
export function coverFit(frame: FrameSize, box: PanelSize): CoverFit {
  if (
    box.width <= 0 ||
    box.height <= 0 ||
    frame.width <= 0 ||
    frame.height <= 0
  ) {
    return { renderW: 0, renderH: 0, offsetX: 0, offsetY: 0, scale: 0 };
  }
  const imgAspect = frame.width / frame.height;
  const boxAspect = box.width / box.height;
  let renderW: number;
  let renderH: number;
  if (boxAspect > imgAspect) {
    // Box is wider than the frame: width fills, height overflows (y crop).
    renderW = box.width;
    renderH = box.width / imgAspect;
  } else {
    // Box is taller/square: height fills, width overflows (x crop).
    renderH = box.height;
    renderW = box.height * imgAspect;
  }
  return {
    renderW,
    renderH,
    offsetX: (box.width - renderW) / 2,
    offsetY: (box.height - renderH) / 2,
    scale: renderW / frame.width,
  };
}

const EPSILON = 1e-9;

/**
 * Normalised frame coordinate (u, v) → panel pixel position.
 *
 * u/v are fractions of the frame, so the position inside the rendered
 * image is (u * frame.width, v * frame.height) frame px, then the cover
 * fit maps it into the panel.
 */
export function framePointToPanel(
  u: number,
  v: number,
  frame: FrameSize,
  box: PanelSize,
  mode: FitMode = "cover",
): { x: number; y: number } {
  const fit = mode === "contain" ? containFit(frame, box) : coverFit(frame, box);
  return { x: fit.offsetX + u * frame.width * fit.scale, y: fit.offsetY + v * frame.height * fit.scale };
}

/**
 * Panel pixel position → normalised frame coordinate. Inverse of
 * `framePointToPanel` within the rendered area; points falling on the
 * letterbox bars clamp to the nearest valid frame edge, so a stray
 * pointer near a border never creates out-of-range geometry.
 */
export function panelPointToFrame(
  x: number,
  y: number,
  frame: FrameSize,
  box: PanelSize,
  mode: FitMode = "cover",
): { u: number; v: number } {
  const fit = mode === "contain" ? containFit(frame, box) : coverFit(frame, box);
  if (fit.scale <= 0) return { u: 0, v: 0 };
  const rawU = (x - fit.offsetX) / (frame.width * fit.scale);
  const rawV = (y - fit.offsetY) / (frame.height * fit.scale);
  return { u: clamp(rawU, 0, 1), v: clamp(rawV, 0, 1) };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Shoelace area of the polygon in normalised units (absolute). */
export function polygonArea(poly: Array<[number, number]>): number {
  if (poly.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

/** Distance between two normalised points expressed in frame px. */
export function lineLengthPx(
  a: [number, number],
  b: [number, number],
  frame: FrameSize,
): number {
  const dx = (b[0] - a[0]) * frame.width;
  const dy = (b[1] - a[1]) * frame.height;
  return Math.hypot(dx, dy);
}

export function samePoint(
  a: [number, number],
  b: [number, number],
  eps: number = 1e-6,
): boolean {
  return Math.abs(a[0] - b[0]) <= eps && Math.abs(a[1] - b[1]) <= eps;
}

export interface ValidationProblem {
  field: "roi" | "in" | "out" | "meta";
  message: string;
}

/**
 * Mirror of the backend validator (`app/mapping.py.validate_mapping`) for
 * the subset the editor can produce, so the operator gets immediate
 * feedback and "Save" is only enabled for payloads the API will accept.
 * The API re-validates; this is never the enforcement mechanism.
 */
export function validateMapping(
  mapping: MappingData,
  frame: FrameSize,
): ValidationProblem[] {
  const problems: ValidationProblem[] = [];

  if (mapping.roi.length < 3) {
    problems.push({ field: "roi", message: "The queue area needs at least 3 points." });
    return problems; // cannot compute area without a polygon
  }
  for (const [u, v] of mapping.roi) {
    if (u < 0 || u > 1 || v < 0 || v > 1) {
      problems.push({ field: "roi", message: "Queue area point is outside the frame." });
      break;
    }
  }
  if (polygonArea(mapping.roi) <= EPSILON) {
    problems.push({ field: "roi", message: "Queue area is degenerate (no area)." });
  }
  if (!isSimplePolygon(mapping.roi)) {
    problems.push({ field: "roi", message: "Queue area edges cross each other." });
  }

  const minLinePx = MIN_LINE_FRACTION_OF_DIAGONAL * Math.hypot(frame.width, frame.height);
  (Object.keys(mapping.lines) as LineId[]).forEach((id) => {
    const { points, direction } = mapping.lines[id];
    const [a, b] = points;
    if (samePoint(a, b)) {
      problems.push({ field: id, message: `The ${id.toUpperCase()} line needs two different points.` });
      return;
    }
    const [au, av] = a;
    const [bu, bv] = b;
    if (au < 0 || au > 1 || av < 0 || av > 1 || bu < 0 || bu > 1 || bv < 0 || bv > 1) {
      problems.push({ field: id, message: `${id.toUpperCase()} line is outside the frame.` });
    }
    const len = lineLengthPx(a, b, frame);
    if (len < minLinePx) {
      problems.push({
        field: id,
        message: `The ${id.toUpperCase()} line is too short (${Math.round(len)} px, needs ${Math.round(minLinePx)}).`,
      });
    }
    if (direction !== "IN" && direction !== "OUT") {
      problems.push({ field: id, message: `Invalid ${id.toUpperCase()} line direction.` });
    }
  });

  if (mapping.lines.in && mapping.lines.out) {
    const ins = mapping.lines.in.points;
    const out = mapping.lines.out.points;
    if (
      samePoint(ins[0], out[0]) &&
      samePoint(ins[1], out[1]) &&
      mapping.lines.in.direction === mapping.lines.out.direction
    ) {
      problems.push({
        field: "out",
        message: "IN and OUT lines are identical — move one of them.",
      });
    }
  }

  return problems;
}

/** Fraction of the frame diagonal below which a line cannot count reliably. */
export const MIN_LINE_FRACTION_OF_DIAGONAL = 0.01;

/**
 * Orientation test for a simple polygon: no two non-adjacent edges cross.
 * Adjacent edges legitimately share a vertex; those are skipped.
 */
export function isSimplePolygon(poly: Array<[number, number]>): boolean {
  const n = poly.length;
  if (n < 3) return false;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if ((j - i) % n === 1 || (i - j + n) % n === 1) continue;
      if (segmentsIntersect(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n])) {
        return false;
      }
    }
  }
  return true;
}

function segmentsIntersect(
  p1: [number, number],
  p2: [number, number],
  p3: [number, number],
  p4: [number, number],
): boolean {
  const o = (a: [number, number], b: [number, number], c: [number, number]) => {
    const v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (Math.abs(v) < 1e-12) return 0;
    return v > 0 ? 1 : -1;
  };
  const onSegment = (a: [number, number], b: [number, number], c: [number, number]) =>
    Math.min(a[0], b[0]) <= c[0] && c[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= c[1] && c[1] <= Math.max(a[1], b[1]);
  const o1 = o(p1, p2, p3);
  const o2 = o(p1, p2, p4);
  const o3 = o(p3, p4, p1);
  const o4 = o(p3, p4, p2);
  if (o1 !== o2 && o3 !== o4) return true;
  if (o1 === 0 && onSegment(p1, p2, p3)) return true;
  if (o2 === 0 && onSegment(p1, p2, p4)) return true;
  if (o3 === 0 && onSegment(p3, p4, p1)) return true;
  if (o4 === 0 && onSegment(p3, p4, p2)) return true;
  return false;
}

/**
 * Deep-clone a mapping (dragging mutates copies, never the live config).
 */
export function cloneMapping(m: MappingData): MappingData {
  return {
    roi: m.roi.map(([u, v]) => [u, v] as [number, number]),
    lines: {
      in: {
        points: [[m.lines.in.points[0][0], m.lines.in.points[0][1]], [m.lines.in.points[1][0], m.lines.in.points[1][1]]],
        direction: m.lines.in.direction,
      },
      out: {
        points: [[m.lines.out.points[0][0], m.lines.out.points[0][1]], [m.lines.out.points[1][0], m.lines.out.points[1][1]]],
        direction: m.lines.out.direction,
      },
    },
    referencePoint: m.referencePoint,
    sourceShape: m.sourceShape ? { ...m.sourceShape } : undefined,
  };
}

/** Structural equality (points and directions) for dirty-state checks. */
export function mappingsEqual(a: MappingData, b: MappingData): boolean {
  const pointsEqual = (x: Array<[number, number]>, y: Array<[number, number]>) =>
    x.length === y.length && x.every(([ux, uy], i) => samePoint([ux, uy], y[i]));
  if (!pointsEqual(a.roi, b.roi)) return false;
  return (["in", "out"] as LineId[]).every((id) => {
    const la = a.lines[id];
    const lb = b.lines[id];
    return (
      samePoint(la.points[0], lb.points[0]) &&
      samePoint(la.points[1], lb.points[1]) &&
      la.direction === lb.direction
    );
  });
}
