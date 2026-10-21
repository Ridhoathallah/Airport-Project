import { describe, it, expect } from "vitest";
import {
  containFit,
  coverFit,
  framePointToPanel,
  panelPointToFrame,
  polygonArea,
  lineLengthPx,
  samePoint,
  validateMapping,
  isSimplePolygon,
  cloneMapping,
  mappingsEqual,
  MappingData,
} from "./mapping";

/** 16:9 frame in a square 200×200 panel: pillarboxed, x-offset = (200-200*(16/9))/2. */
const FRAME_16x9 = { width: 1280, height: 720 };
const SQUARE_PANEL = { width: 200, height: 200 };

/** 16:9 frame in a 16:9 panel: exact fit, no letterbox. */
const PANEL_16x9 = { width: 160, height: 90 };

describe("object-cover fit", () => {
  it("fills a matching-aspect panel without offsets", () => {
    const fit = coverFit(FRAME_16x9, PANEL_16x9);
    expect(fit.renderW).toBeCloseTo(160);
    expect(fit.renderH).toBeCloseTo(90);
    expect(fit.offsetX).toBeCloseTo(0);
    expect(fit.offsetY).toBeCloseTo(0);
    expect(fit.scale).toBeCloseTo(160 / 1280);
  });

  it("pillarboxes in a squarer panel (equal crops both sides)", () => {
    const fit = coverFit(FRAME_16x9, SQUARE_PANEL);
    // height fills the box; width overflows and is cropped equally
    expect(fit.renderH).toBeCloseTo(200);
    expect(fit.renderW).toBeCloseTo(200 * (1280 / 720));
    expect(fit.offsetY).toBeCloseTo(0);
    expect(fit.offsetX).toBeLessThan(0); // cropped left bar
    // overflow is split equally: |offsetX| === (renderW - boxW) / 2
    expect(-fit.offsetX).toBeCloseTo((fit.renderW - SQUARE_PANEL.width) / 2, 6);
    expect(fit.scale).toBeCloseTo(fit.renderW / FRAME_16x9.width);
  });

  it("crops top/bottom in a wider panel (16:9 frame, 21:9 box)", () => {
    const box = { width: 400, height: 140 }; // aspect 2.857 > 1.777
    const fit = coverFit(FRAME_16x9, box);
    expect(fit.renderW).toBeCloseTo(400);
    expect(fit.renderH).toBeCloseTo(400 / (1280 / 720));
    expect(fit.renderH).toBeGreaterThan(box.height); // overflow cropped
    expect(fit.offsetX).toBeCloseTo(0);
    expect(fit.offsetY).toBeLessThan(0); // equal split above and below
    expect(-fit.offsetY).toBeCloseTo((fit.renderH - box.height) / 2, 6);
  });

  it("scales with tiny and zero panels without crashing", () => {
    expect(coverFit(FRAME_16x9, { width: 0, height: 0 })).toEqual({
      renderW: 0,
      renderH: 0,
      offsetX: 0,
      offsetY: 0,
      scale: 0,
    });
    const fit = coverFit({ width: 0, height: 0 }, SQUARE_PANEL);
    expect(fit.scale).toBe(0);
  });

  it("resizes correctly when the panel grows" /* regression: overlay drift */, () => {
    const small = coverFit(FRAME_16x9, { width: 100, height: 50 });
    const large = coverFit(FRAME_16x9, { width: 200, height: 100 });
    // same aspect boxes: the geometry scales linearly with panel size
    expect(large.scale).toBeCloseTo(small.scale * 2);
    expect(framePointToPanel(0.5, 0.5, FRAME_16x9, { width: 100, height: 50 }).y).toBeGreaterThan(0);
  });
});

describe("letterbox (contain) fit — the mapping editor's view", () => {
  it("shows the whole frame with equal bars on both sides", () => {
    const fit = containFit(FRAME_16x9, SQUARE_PANEL);
    // the frame's larger dimension fills; the other is letterboxed evenly
    expect(fit.renderW).toBeCloseTo(200);
    expect(fit.renderH).toBeCloseTo(200 / (1280 / 720));
    expect(fit.offsetX).toBeCloseTo(0);
    expect(fit.offsetY).toBeCloseTo((SQUARE_PANEL.height - fit.renderH) / 2, 6);
    expect(fit.scale).toBeCloseTo(fit.renderW / FRAME_16x9.width);
  });

  it("every frame corner maps INSIDE the panel (all handles reachable)", () => {
    const box = { width: 791, height: 225 }; // the very short panel shape
    for (const [u, v] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
      const p = framePointToPanel(u, v, FRAME_16x9, box, "contain");
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x).toBeLessThanOrEqual(box.width);
      expect(p.y).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeLessThanOrEqual(box.height);
    }
  });

  it("round-trips corner vertices exactly", () => {
    const box = { width: 791, height: 225 };
    for (const [u, v] of [[0, 0], [1, 0.5], [0.5, 1], [0.25, 0.75]] as const) {
      const p = framePointToPanel(u, v, FRAME_16x9, box, "contain");
      const back = panelPointToFrame(p.x, p.y, FRAME_16x9, box, "contain");
      expect(back.u).toBeCloseTo(u, 6);
      expect(back.v).toBeCloseTo(v, 6);
    }
  });

  it("bars clamp stray pointers to the frame edge", () => {
    const box = { width: 791, height: 225 };
    // x at the panel centre; y far above the panel → clamped to v=0
    const stray = panelPointToFrame(box.width / 2, -500, FRAME_16x9, box, "contain");
    expect(stray.v).toBe(0);
    expect(stray.u).toBeCloseTo(0.5, 6);
  });
});

describe("frame ↔ panel pointer mapping", () => {
  it("round-trips centre through a letterboxed panel", () => {
    const fit = coverFit(FRAME_16x9, SQUARE_PANEL);
    const centre = framePointToPanel(0.5, 0.5, FRAME_16x9, SQUARE_PANEL);
    // centre of the frame = centre of the rendered area = centre of panel
    expect(centre.x).toBeCloseTo(fit.offsetX + (fit.renderW / 2), 6);
    expect(centre.y).toBeCloseTo(SQUARE_PANEL.height / 2, 6);
    const back = panelPointToFrame(centre.x, centre.y, FRAME_16x9, SQUARE_PANEL);
    expect(back.u).toBeCloseTo(0.5, 6);
    expect(back.v).toBeCloseTo(0.5, 6);
  });

  it("round-trips an off-centre point exactly", () => {
    for (const [u, v] of [[0.1, 0.2], [0.9, 0.8], [0.25, 0.75]]) {
      const p = framePointToPanel(u, v, FRAME_16x9, SQUARE_PANEL);
      const back = panelPointToFrame(p.x, p.y, FRAME_16x9, SQUARE_PANEL);
      expect(back.u).toBeCloseTo(u, 6);
      expect(back.v).toBeCloseTo(v, 6);
    }
  });

  it("round-trips under letterboxing (21:9 box)", () => {
    const box = { width: 400, height: 140 };
    for (const [u, v] of [[0.5, 0.25], [0.05, 0.9]]) {
      const p = framePointToPanel(u, v, FRAME_16x9, box);
      const back = panelPointToFrame(p.x, p.y, FRAME_16x9, box);
      expect(back.u).toBeCloseTo(u, 6);
      expect(back.v).toBeCloseTo(v, 6);
    }
  });

  it("maps panel edges into interior frame coordinates under cropping", () => {
    // With cover the frame OVERFLOWS and is cropped: a panel edge is
    // inside the rendered image, so it maps to an interior v, not 0/1.
    const box = { width: 400, height: 140 };
    const fit = coverFit(FRAME_16x9, box);
    const top = panelPointToFrame(200, 0, FRAME_16x9, box);
    expect(top.u).toBeCloseTo(0.5, 6);
    expect(top.v).toBeCloseTo(-fit.offsetY / (FRAME_16x9.height * fit.scale), 6);
    expect(top.v).toBeGreaterThan(0);
    expect(top.v).toBeLessThan(0.5);
    const left = panelPointToFrame(0, 70, FRAME_16x9, box);
    expect(left.v).toBeCloseTo(0.5, 6);
    expect(left.u).toBeGreaterThanOrEqual(0);
    expect(left.u).toBeLessThan(0.5);
  });

  it("never produces out-of-range coordinates for stray pointers", () => {
    // extreme pointers (corner, outside crop) clamp inside [0,1]
    const stray = panelPointToFrame(-500, 9999, FRAME_16x9, SQUARE_PANEL);
    expect(stray.u).toBeGreaterThanOrEqual(0);
    expect(stray.u).toBeLessThanOrEqual(1);
    expect(stray.v).toBeLessThanOrEqual(1);
  });

  it("degenerate sizes map to origin", () => {
    expect(panelPointToFrame(10, 10, FRAME_16x9, { width: 0, height: 0 })).toEqual({ u: 0, v: 0 });
  });
});

describe("geometry helpers", () => {
  it("polygonArea computes the shoelace area", () => {
    const unitSquare: Array<[number, number]> = [[0, 0], [1, 0], [1, 1], [0, 1]];
    expect(polygonArea(unitSquare)).toBeCloseTo(1);
    expect(polygonArea([[0, 0], [1, 0], [1, 1]])).toBeCloseTo(0.5);
    expect(polygonArea([[0, 0], [1, 0]])).toBe(0);
  });

  it("lineLengthPx uses frame pixels, not normalised units", () => {
    expect(lineLengthPx([0.25, 0.5], [0.75, 0.5], FRAME_16x9)).toBeCloseTo(0.5 * 1280);
    expect(lineLengthPx([0.25, 0.25], [0.25, 0.75], FRAME_16x9)).toBeCloseTo(0.5 * 720);
  });

  it("samePoint honours its epsilon", () => {
    expect(samePoint([0.5, 0.5], [0.5, 0.5])).toBe(true);
    expect(samePoint([0.5, 0.5], [0.5, 0.50000001])).toBe(true); // 1e-8: inside 1e-6
    expect(samePoint([0.5, 0.5], [0.5000005, 0.5])).toBe(true); // 5e-7: inside
    expect(samePoint([0.5, 0.5], [0.50001, 0.5])).toBe(false);
  });

  it("isSimplePolygon rejects a bowtie", () => {
    expect(isSimplePolygon([[0, 0], [1, 1], [1, 0], [0, 1]])).toBe(false);
    expect(isSimplePolygon([[0, 0], [1, 0], [1, 1], [0, 1]])).toBe(true);
  });
});

function makeMapping(): MappingData {
  return {
    roi: [[0.125, 0.1389], [0.875, 0.1389], [0.875, 0.7222], [0.125, 0.7222]],
    lines: {
      in: { points: [[0.875, 0.7778], [0.125, 0.7778]], direction: "IN" },
      out: { points: [[0.125, 0.7778], [0.875, 0.7778]], direction: "OUT" },
    },
    referencePoint: "bottom_center",
  };
}

describe("validateMapping", () => {
  it("accepts the shipped configuration", () => {
    expect(validateMapping(makeMapping(), FRAME_16x9)).toEqual([]);
  });

  it("rejects a two-vertex ROI", () => {
    const m = makeMapping();
    m.roi = [[0.1, 0.1], [0.9, 0.1]];
    const problems = validateMapping(m, FRAME_16x9);
    expect(problems.some((p) => p.field === "roi" && /3 points/.test(p.message))).toBe(true);
  });

  it("rejects an out-of-frame ROI vertex", () => {
    const m = makeMapping();
    m.roi = [[0.1, 0.1], [1.4, 0.1], [0.9, 0.8], [0.1, 0.8]];
    expect(validateMapping(m, FRAME_16x9).some((p) => /outside the frame/.test(p.message))).toBe(true);
  });

  it("rejects a self-crossing ROI", () => {
    const m = makeMapping();
    m.roi = [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]];
    const problems = validateMapping(m, FRAME_16x9);
    expect(problems.some((p) => /cross each other/.test(p.message))).toBe(true);
  });

  it("rejects a degenerate zero-area ROI", () => {
    const m = makeMapping();
    m.roi = [[0.1, 0.1], [0.4, 0.1], [0.7, 0.1], [0.9, 0.1]];
    expect(validateMapping(m, FRAME_16x9).some((p) => /degenerate/.test(p.message))).toBe(true);
  });

  it("rejects identical line endpoints", () => {
    const m = makeMapping();
    m.lines.in = { points: [[0.5, 0.5], [0.5, 0.5]], direction: "IN" };
    expect(validateMapping(m, FRAME_16x9).some((p) => p.field === "in" && /different points/.test(p.message))).toBe(true);
  });

  it("rejects a line shorter than 1% of the frame diagonal", () => {
    const m = makeMapping();
    // diagonal ≈ 1468.6 px → floor ≈ 14.7 px → 5 px fails
    m.lines.in = { points: [[0.5, 0.5], [0.504, 0.5]], direction: "IN" };
    const problems = validateMapping(m, FRAME_16x9);
    expect(problems.some((p) => p.field === "in" && /too short/.test(p.message))).toBe(true);
  });

  it("rejects identical IN and OUT lines", () => {
    const m = makeMapping();
    m.lines.in = { points: [[0.1, 0.9], [0.9, 0.9]], direction: "OUT" };
    m.lines.out = { points: [[0.1, 0.9], [0.9, 0.9]], direction: "OUT" };
    expect(validateMapping(m, FRAME_16x9).some((p) => /identical/.test(p.message))).toBe(true);
  });

  it("allows IN/OUT lines sharing geometry with opposite directions", () => {
    const m = makeMapping();
    m.lines.in = { points: [[0.1, 0.9], [0.9, 0.9]], direction: "IN" };
    m.lines.out = { points: [[0.1, 0.9], [0.9, 0.9]], direction: "OUT" };
    expect(validateMapping(m, FRAME_16x9)).toEqual([]);
  });
});

describe("clone + dirty-state equality", () => {
  it("clone produces an independent deep copy", () => {
    const m = makeMapping();
    const copy = cloneMapping(m);
    copy.roi[0][0] = 0.999;
    copy.lines.in.points[0][1] = 0.123;
    expect(m.roi[0][0]).toBe(0.125);
    expect(m.lines.in.points[0][1]).toBeCloseTo(0.7778);
  });

  it("mappingsEqual is false for any real edit and true after revert", () => {
    const a = makeMapping();
    const b = cloneMapping(a);
    expect(mappingsEqual(a, b)).toBe(true);
    b.roi[3][0] = 0.15;
    expect(mappingsEqual(a, b)).toBe(false);
    b.roi[3][0] = a.roi[3][0];
    b.lines.out.points[0][1] = 0.8;
    expect(mappingsEqual(a, b)).toBe(false);
    b.lines.out.points[0][1] = a.lines.out.points[0][1];
    expect(mappingsEqual(a, b)).toBe(true);
  });
});
