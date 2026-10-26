import { describe, it, expect } from "vitest";
import {
  convertApiSnapshot,
  deriveConnection,
  createUnavailableSnapshot,
  operatorText,
  describeEvent,
} from "./telemetry";
import type { SnapshotResponse } from "./api";

/** A full contract-shaped snapshot: every required wire field is real. */
function makeResponse(overrides: Partial<SnapshotResponse> = {}): SnapshotResponse {
  const base: SnapshotResponse = {
    scenario: "ONLINE",
    queue: {
      queue: 12,
      capacity: 30,
      occupancy: 40,
      congestion_sec: 0,
      entry_rate: 3,
      exit_rate: 2,
      net_change: 1,
      trend: "GROWING",
      avg_wait: 1.5,
      longest_wait: 4,
      throughput: 42,
      state: "CROWDED",
      state_active: true,
      reason: "Elevated demand",
      action: "Open a second lane",
      predicted_wait: 3,
      clearance_time: 10,
      service_time_s: 18.5,
      pacing_status: "STEADY",
    },
    system: {
      inference_ms: 25.5,
      model: "YOLOv5s",
      tracker: "SORT",
      detections: 7,
      active_tracks: 6,
      detection_status: "OK",
      tracking_status: "OK",
      analytics_status: "OK",
      device: "mps",
    },
    camera: {
      link: "ONLINE",
      status: "OK",
      fps: 24,
      resolution: "1920x1080",
      latency_ms: null,
      packet_loss: null,
      last_frame_age_ms: 48,
      uptime_sec: 123,
      source_label: "CAM-01",
      source_index: null,
      rotation_deg: 0,
    },
    recording: { active: false, since_sec: null, size_gb: null },
    last_known_queue: 12,
    last_update: 1700000000000,
    history: [
      { t: 1700000000000, queue: 12, occupancy: 40, entry_rate: 3, exit_rate: 2, avg_wait: 1.5 },
    ],
    events: [
      {
        id: 1,
        timestamp: 1700000000000,
        kind: "entry",
        severity: "info",
        label: "Passenger entered",
        detail: "18 -> 19",
      },
    ],
    tracks: [
      { id: 1, x: 0.5, y: 0.5, w: 0.1, h: 0.2, in_roi: true, confidence: 0.9, age: 10, hits: 8, state: "active" },
    ],
    frame_ts: 1700000000000,
  };
  return { ...base, ...overrides };
}

describe("convertApiSnapshot", () => {
  it("passes the classifier state through unchanged", () => {
    const snap = convertApiSnapshot(makeResponse());
    expect(snap.state).toBe("CROWDED");
    expect(snap.connection).toBe("LIVE");
    expect(snap.stateActive).toBe(true);
    expect(snap.lastKnownQueue).toBe(12);
    expect(snap.camera.sourceLabel).toBe("CAM-01");
  });

  it("maps measured values, including minute units on the wire", () => {
    const snap = convertApiSnapshot(makeResponse());
    expect(snap.queue).toBe(12);
    expect(snap.occupancy).toBe(40);
    expect(snap.predictedWait).toBe(3);
    expect(snap.clearanceTime).toBe(10);
    expect(snap.serviceTimeS).toBe(18.5);
    expect(snap.pacingStatus).toBe("STEADY");
    expect(snap.history).toHaveLength(1);
    expect(snap.history[0].entryRate).toBe(3);
    expect(snap.events[0].label).toBe("Passenger entered");
    expect(snap.tracks[0].inRoi).toBe(true);
  });

  it("keeps null measurements null instead of inventing zeros", () => {
    const q = makeResponse().queue;
    const snap = convertApiSnapshot(
      makeResponse({
        queue: {
          ...q,
          queue: null,
          occupancy: null,
          entry_rate: null,
          exit_rate: null,
          net_change: null,
          trend: null,
          avg_wait: null,
          longest_wait: null,
          throughput: null,
          predicted_wait: null,
          clearance_time: null,
          service_time_s: null,
          pacing_status: null,
        },
      }),
    );
    expect(snap.queue).toBeNull();
    expect(snap.occupancy).toBeNull();
    expect(snap.entryRate).toBeNull();
    expect(snap.exitRate).toBeNull();
    expect(snap.netChange).toBeNull();
    expect(snap.trend).toBeNull();
    expect(snap.avgWait).toBeNull();
    expect(snap.throughput).toBeNull();
    expect(snap.predictedWait).toBeNull();
    expect(snap.clearanceTime).toBeNull();
    expect(snap.serviceTimeS).toBeNull();
    expect(snap.pacingStatus).toBeNull();
  });

  it("treats a missing source label as unreported, not as a camera name", () => {
    const snap = convertApiSnapshot(
      makeResponse({ camera: { ...makeResponse().camera, source_label: "" } }),
    );
    expect(snap.camera.sourceLabel).toBe("");
  });
});

describe("deriveConnection", () => {
  it("reads the backend's availability summary", () => {
    expect(deriveConnection(makeResponse({ scenario: "ONLINE" }))).toBe("LIVE");
    expect(deriveConnection(makeResponse({ scenario: "DEGRADED" }))).toBe("DEGRADED");
    expect(deriveConnection(makeResponse({ scenario: "NO_DATA" }))).toBe("NO_DATA");
    expect(deriveConnection(makeResponse({ scenario: "OFFLINE" }))).toBe("CAMERA_OFFLINE");
  });

  it("lets the capture supervisor's status outrank the link summary", () => {
    const off = makeResponse().camera;
    expect(
      deriveConnection(
        makeResponse({ scenario: "OFFLINE", camera: { ...off, link: "OFFLINE", status: "RECONNECTING" } }),
      ),
    ).toBe("RECONNECTING");
    expect(
      deriveConnection(
        makeResponse({ scenario: "OFFLINE", camera: { ...off, link: "OFFLINE", status: "NO_FRAMES" } }),
      ),
    ).toBe("NO_FRAMES");
  });
});

describe("createUnavailableSnapshot", () => {
  it("reports every measurement as null, never as zero", () => {
    const snap = createUnavailableSnapshot("CAMERA_OFFLINE", "Camera unavailable");
    expect(snap.state).toBe("DATA_UNAVAILABLE");
    expect(snap.stateActive).toBe(false);
    expect(snap.connection).toBe("CAMERA_OFFLINE");
    expect(snap.reason).toBe("Camera unavailable");
    expect(snap.queue).toBeNull();
    expect(snap.occupancy).toBeNull();
    expect(snap.entryRate).toBeNull();
    expect(snap.avgWait).toBeNull();
    expect(snap.netChange).toBeNull();
    expect(snap.lastKnownQueue).toBeNull();
    expect(snap.lastUpdate).toBe(0);
    expect(snap.history).toHaveLength(0);
    expect(snap.events).toHaveLength(0);
    expect(snap.tracks).toHaveLength(0);
  });

  it("carries connectivity detail without fabricating an identifier", () => {
    const snap = createUnavailableSnapshot("RECONNECTING", "Retrying");
    expect(snap.camera.status).toBe("RECONNECTING");
    expect(snap.camera.sourceLabel).toBe("");
    expect(snap.camera.sourceIndex).toBeNull();
    expect(snap.camera.link).toBe("OFFLINE");
    expect(snap.cv.detection).toBe("STALLED");
    expect(snap.cv.tracking).toBe("STALLED");
    expect(snap.cv.analytics).toBe("STALLED");
  });
});

describe("operator-facing text", () => {
  it("rewrites internal state names to operator words", () => {
    expect(operatorText("State changed to CROWDED")).toBe("State changed to Busy");
    expect(operatorText("CONGESTED to BACKLOG")).toBe("Congested to Backlog");
    expect(operatorText("geometry UNCONFIGURED")).toBe("geometry Setup incomplete");
    expect(operatorText("Nothing to rewrite")).toBe("Nothing to rewrite");
  });

  it("cleans both label and detail of an event", () => {
    const { label, detail } = describeEvent({
      id: 1,
      t: 0,
      kind: "state",
      severity: "info",
      label: "State changed to CROWDED",
      detail: "NORMAL to CROWDED",
    });
    expect(label).toBe("State changed to Busy");
    expect(detail).toBe("Normal to Busy");
  });
});
