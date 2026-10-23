import { useEffect, useRef, useState, useCallback } from "react";
import { getSnapshot, getConfig, type SnapshotResponse, type ConfigResponse } from "./api";

// ─────────────────────────────────────────────────────────────────────────
// Camera-only queue telemetry simulator.
// Single source of truth: every derived metric (occupancy, rates, wait,
// state, trend) is computed from the same underlying queue dynamics so the
// numbers always logically agree with each other.
// ─────────────────────────────────────────────────────────────────────────

export type QueueState = "NORMAL" | "BUSY" | "CONGESTED" | "BACKLOG";
export type Scenario = QueueState | "OFFLINE" | "DEGRADED";
export type PipelineStatus = "OK" | "DEGRADED" | "STALLED";
export type LinkStatus = "ONLINE" | "OFFLINE" | "DEGRADED";

export interface Sample {
  t: number; // epoch ms
  queue: number;
  occupancy: number; // %
  entryRate: number; // per min
  exitRate: number; // per min
  avgWait: number; // min
}

export interface CameraHealth {
  link: LinkStatus;
  fps: number;
  resolution: string;
  latencyMs: number | null; // null = not measured by this pipeline
  packetLoss: number | null; // %  null = not measured by this pipeline
  lastFrameAgeMs: number;
  uptimeSec: number;
  /** Camera chrome label reported by the pipeline, e.g. "CAM-01". */
  sourceLabel: string;
  /** Live webcam index; null for file/IP/image sources. */
  sourceIndex: number | null;
  /** Clockwise rotation applied at the capture edge (degrees). */
  rotationDeg: number;
}

export interface CvHealth {
  detection: PipelineStatus;
  tracking: PipelineStatus;
  analytics: PipelineStatus;
  inferenceMs: number;
  model: string;
  tracker: string;
  detections: number;
  activeTracks: number;
  /** Compute device the inference ran on ("mps", "cpu", ...). */
  device: string;
}

export interface QueueEvent {
  id: number;
  t: number;
  kind: "entry" | "exit" | "state" | "congestion" | "system";
  severity: "info" | "warn" | "critical" | "good";
  label: string;
  detail?: string; // operator-facing change, e.g. "18 → 19" or "Normal to Busy"
  /** Engineering description of the same event, shown to technicians only. */
  technical?: string;
}

/**
 * Internal detection states, mapped to the words an operator reads.
 *
 * The detector classifies the queue as EMPTY / SPARSE / NORMAL / CROWDED /
 * CONGESTED / BACKLOG / UNCONFIGURED. Those names are engineering vocabulary
 * and never reach the operator surface: the dashboard shows Normal, Busy,
 * Congested and Backlog, and reports an incomplete setup as such.
 */
const STATE_WORDS: Record<string, string> = {
  EMPTY: "Empty",
  SPARSE: "Quiet",
  NORMAL: "Normal",
  BUSY: "Busy",
  CROWDED: "Busy",
  CONGESTED: "Congested",
  BACKLOG: "Backlog",
  UNCONFIGURED: "Setup incomplete",
  DATA_UNAVAILABLE: "No data",
};

/** Rewrite any internal state name inside a piece of operator-facing text. */
export function operatorText(text: string): string {
  return Object.entries(STATE_WORDS).reduce(
    (acc, [internal, word]) => acc.replace(new RegExp(`\\b${internal}\\b`, "g"), word),
    text,
  );
}

/**
 * The label and detail an operator actually reads for an event.
 *
 * The backend already emits operator language; this is the safety net that
 * keeps an implementation name out of the activity log if one ever slips
 * through, and it normalises the simulator in the same way.
 */
export function describeEvent(e: QueueEvent): { label: string; detail?: string } {
  const label = operatorText(e.label);
  const detail = e.detail ? operatorText(e.detail) : undefined;
  return { label, detail };
}

export interface Track {
  id: number;
  x: number; // 0..1 within frame
  y: number;
  w: number;
  h: number;
  inRoi: boolean;
}

export interface Snapshot {
  scenario: Scenario;
  state: QueueState;
  stateActive: boolean; // false when OFFLINE — queue reading not trustworthy
  reason: string;
  queue: number;
  capacity: number;
  occupancy: number;
  congestionSec: number;
  entryRate: number;
  exitRate: number;
  netChange: number; // per min
  trend: "GROWING" | "CLEARING" | "STEADY";
  avgWait: number; // min
  longestWait: number; // min
  throughput: number; // per hr
  /** Operational guidance for the current condition, in operator language. */
  action?: string;
  predictedWait?: number; // min (Little's Law queue wait estimation)
  clearanceTime?: number; // min (estimated queue clearance time)
  serviceTimeS?: number; // seconds per passenger
  pacingStatus?: "OPTIMAL" | "STEADY" | "SLOW_INSPECTION" | "STAGNANT" | "IDLE";
  camera: CameraHealth;
  cv: CvHealth;
  recording: { active: boolean; sinceSec: number | null; sizeGb: number | null };
  lastKnownQueue: number; // last trustworthy reading (for OFFLINE)
  lastUpdate: number; // epoch ms of last live sample
  history: Sample[];
  events: QueueEvent[];
  tracks: Track[];
  frameTs: number;
}

// The capacity mirrors config/camera.yaml (queue_capacity: 30) so the
// simulator and the live backend agree on what "60% full" means.
const CAPACITY = 30;
const TICK_MS = 1000;
const SIM_STEP_SEC = 15; // each real tick advances simulated time by 15s
const HISTORY_LEN = 240; // 240 * 15s = 60 min

interface Target {
  entry: number;
  exit: number;
  queue: number;
}

// Queue levels are chosen against the live occupancy ladder
// (sparse 10%, normal 40%, crowded 60%, congested 80%) so each scenario
// lands on the state it names.
const TARGETS: Record<QueueState, Target> = {
  NORMAL: { entry: 6, exit: 7, queue: 8 },
  BUSY: { entry: 12, exit: 10.5, queue: 21 },
  CONGESTED: { entry: 15, exit: 9.5, queue: 26 },
  BACKLOG: { entry: 16, exit: 8, queue: 38 },
};

function classify(occupancy: number): QueueState {
  if (occupancy >= 100) return "BACKLOG";
  if (occupancy >= 80) return "CONGESTED";
  if (occupancy >= 60) return "BUSY";
  return "NORMAL";
}

function noise(mag: number) {
  return (Math.random() - 0.5) * 2 * mag;
}

interface SimState {
  queue: number;
  entry: number;
  exit: number;
  congestionSince: number | null;
  history: Sample[];
  events: QueueEvent[];
  eventId: number;
  lastState: QueueState;
  uptimeSec: number;
  recSinceSec: number;
  now: number;
}

function seed(scenario: Scenario): SimState {
  const base: QueueState =
    scenario === "OFFLINE" || scenario === "DEGRADED" ? "NORMAL" : scenario;
  const tgt = TARGETS[base];
  const now = Date.now();
  const history: Sample[] = [];
  let q = tgt.queue;
  for (let i = HISTORY_LEN; i > 0; i--) {
    const t = now - i * SIM_STEP_SEC * 1000;
    q = Math.max(0, q + noise(1.2));
    const occ = (q / CAPACITY) * 100;
    const entry = Math.max(0, tgt.entry + noise(2));
    const exit = Math.max(0.5, tgt.exit + noise(2));
    history.push({
      t,
      queue: Math.round(q),
      occupancy: +occ.toFixed(1),
      entryRate: +entry.toFixed(1),
      exitRate: +exit.toFixed(1),
      avgWait: +(q / Math.max(exit, 0.5)).toFixed(1),
    });
  }
  return {
    queue: tgt.queue,
    entry: tgt.entry,
    exit: tgt.exit,
    congestionSince: base === "CONGESTED" || base === "BACKLOG" ? now - 214000 : null,
    history,
    events: seedEvents(now),
    eventId: 100,
    lastState: base,
    uptimeSec: 27834,
    recSinceSec: 27810,
    now,
  };
}

function seedEvents(now: number): QueueEvent[] {
  return [
    { id: 1, t: now - 44000, kind: "entry", severity: "info", label: "Passenger entered queue" },
    { id: 2, t: now - 96000, kind: "exit", severity: "info", label: "Passenger cleared screening" },
    { id: 3, t: now - 168000, kind: "system", severity: "good", label: "Monitoring active" },
    { id: 4, t: now - 240000, kind: "state", severity: "info", label: "Queue status changed" },
  ];
}

function reasonFor(state: QueueState, net: number, occ: number): string {
  if (state === "BACKLOG")
    return `Queue exceeds capacity (${occ.toFixed(0)}%): entries outpacing exits`;
  if (state === "CONGESTED")
    return net > 1
      ? `Occupancy above 80% and rising: entry rate exceeds exit rate`
      : `Occupancy above 80%: throughput below demand`;
  if (state === "BUSY")
    return net > 1 ? `Elevated demand: queue filling` : `Elevated occupancy, flow balanced`;
  return net < -1 ? `Flow clearing normally` : `Nominal: occupancy within safe range`;
}

/**
 * Operational guidance, mirroring the recommendation table the live
 * pipeline uses (queue_prototype.DEFAULT_RECOMMENDATIONS) so the simulated
 * console gives the same guidance as the real one.
 */
const SIM_ACTIONS: Record<QueueState, string> = {
  NORMAL: "Normal load. Monitor lane throughput; no action required.",
  BUSY: "Queue building. Consider opening an additional screening lane.",
  CONGESTED:
    "Congestion detected. Activate overflow management and direct passengers to secondary lanes.",
  BACKLOG: "Backlog: flow below baseline with long waits. Activate the queue-management protocol.",
};

function step(s: SimState, scenario: Scenario): SimState {
  const offline = scenario === "OFFLINE";
  const degraded = scenario === "DEGRADED";
  s.now += SIM_STEP_SEC * 1000;
  s.uptimeSec += SIM_STEP_SEC;
  if (offline) {
    // frozen queue reading; nothing trustworthy updates
    return s;
  }
  s.recSinceSec += SIM_STEP_SEC;
  const base = scenario === "DEGRADED" ? s.lastState : (scenario as QueueState);
  const tgt = TARGETS[base] ?? TARGETS.NORMAL;

  // ease rates toward target with noise
  s.entry += (tgt.entry - s.entry) * 0.25 + noise(1.5);
  s.exit += (tgt.exit - s.exit) * 0.25 + noise(1.2);
  s.entry = Math.max(0, s.entry);
  s.exit = Math.max(0.4, s.exit);

  const net = (s.entry - s.exit) * (SIM_STEP_SEC / 60);
  // pull queue toward target so scenario stays coherent, plus flow + noise
  s.queue += net + (tgt.queue - s.queue) * 0.12 + noise(0.8);
  s.queue = Math.max(0, s.queue);

  const prevQueue = Math.round(s.history[s.history.length - 1]?.queue ?? s.queue);
  const occ = (s.queue / CAPACITY) * 100;
  const state = classify(occ);
  const nowQueue = Math.round(s.queue);

  // congestion timer
  if (occ >= 80) {
    if (s.congestionSince == null) s.congestionSince = s.now;
  } else {
    s.congestionSince = null;
  }

  // events on state change — operator-facing language, no engineering names
  if (state !== s.lastState) {
    const sev =
      state === "BACKLOG"
        ? "critical"
        : state === "CONGESTED"
          ? "warn"
          : state === "NORMAL"
            ? "good"
            : "info";
    s.events.unshift({
      id: ++s.eventId,
      t: s.now,
      kind: "state",
      severity: sev as QueueEvent["severity"],
      label: "Queue status changed",
      detail: `${s.lastState} → ${state}`,
    });
    if ((state === "CONGESTED" || state === "BACKLOG") && s.lastState !== "BACKLOG") {
      s.events.unshift({
        id: ++s.eventId,
        t: s.now,
        kind: "congestion",
        severity: state === "BACKLOG" ? "critical" : "warn",
        label: state === "BACKLOG" ? "Queue backlog forming" : "Congestion detected",
        detail: `${occ.toFixed(0)}% occupancy`,
      });
    }
    if ((state === "BUSY" || state === "NORMAL") && (s.lastState === "CONGESTED" || s.lastState === "BACKLOG")) {
      s.events.unshift({
        id: ++s.eventId,
        t: s.now,
        kind: "congestion",
        severity: "good",
        label: "Queue beginning to clear",
      });
    }
    s.lastState = state;
  }

  // occasional passenger flow events with queue-count transition
  if (Math.random() < 0.5 && nowQueue !== prevQueue) {
    const entering = nowQueue > prevQueue;
    s.events.unshift({
      id: ++s.eventId,
      t: s.now,
      kind: entering ? "entry" : "exit",
      severity: "info",
      label: entering ? "Passenger entered queue" : "Passenger cleared screening",
      detail: `${prevQueue} → ${nowQueue}`,
    });
  }
  if (degraded && Math.random() < 0.15) {
    s.events.unshift({
      id: ++s.eventId,
      t: s.now,
      kind: "system",
      severity: "warn",
      label: "Camera image quality reduced",
    });
  }

  if (s.events.length > 60) s.events.length = 60;

  s.history.push({
    t: s.now,
    queue: Math.round(s.queue),
    occupancy: +occ.toFixed(1),
    entryRate: +s.entry.toFixed(1),
    exitRate: +s.exit.toFixed(1),
    avgWait: +(s.queue / Math.max(s.exit, 0.5)).toFixed(1),
  });
  if (s.history.length > HISTORY_LEN) s.history.shift();
  return s;
}

function tracksFor(queue: number, degraded: boolean): Track[] {
  const n = Math.min(12, Math.max(0, Math.round(queue / 4)));
  const out: Track[] = [];
  for (let i = 0; i < n; i++) {
    const col = i % 3;
    const row = Math.floor(i / 3);
    out.push({
      id: 2200 + i,
      x: 0.18 + col * 0.22 + noise(0.02),
      y: 0.26 + row * 0.17 + noise(0.02),
      w: 0.11,
      h: 0.2,
      inRoi: !degraded || Math.random() > 0.3,
    });
  }
  return out;
}

function build(s: SimState, scenario: Scenario): Snapshot {
  const offline = scenario === "OFFLINE";
  const degraded = scenario === "DEGRADED";
  const last = s.history[s.history.length - 1];
  const occ = last.occupancy;
  const state = classify(occ);
  const net = +(last.entryRate - last.exitRate).toFixed(1);
  const trend = net > 1 ? "GROWING" : net < -1 ? "CLEARING" : "STEADY";
  const congestionSec = s.congestionSince ? Math.round((s.now - s.congestionSince) / 1000) : 0;
  const avgWait = last.avgWait;
  const longestWait = +(avgWait * 1.7 + 0.8).toFixed(1);
  const serviceTimeS = +(60 / Math.max(last.exitRate, 0.5)).toFixed(1);
  const predictedWait = +(last.queue * (serviceTimeS / 60)).toFixed(1);
  const clearanceTime = predictedWait;
  const pacingStatus =
    last.queue === 0
      ? "IDLE"
      : serviceTimeS <= 15
        ? "OPTIMAL"
        : serviceTimeS > 35
          ? "SLOW_INSPECTION"
          : "STEADY";

  const camera: CameraHealth = offline
    ? {
        link: "OFFLINE",
        fps: 0,
        resolution: "1920×1080",
        latencyMs: 0,
        packetLoss: 100,
        lastFrameAgeMs: 8400,
        uptimeSec: s.uptimeSec,
        sourceLabel: "CAM-01",
        sourceIndex: 0,
        rotationDeg: 0,
      }
    : {
        link: degraded ? "DEGRADED" : "ONLINE",
        fps: degraded ? +(11 + noise(3)).toFixed(0) : +(24 + noise(1)).toFixed(0),
        resolution: "1920×1080",
        latencyMs: degraded ? Math.round(180 + noise(60)) : Math.round(62 + noise(18)),
        packetLoss: degraded ? +(3.4 + noise(1)).toFixed(1) : +(0.1 + Math.random() * 0.2).toFixed(1),
        lastFrameAgeMs: degraded ? Math.round(280 + noise(120)) : Math.round(48 + noise(20)),
        uptimeSec: s.uptimeSec,
        sourceLabel: "CAM-01",
        sourceIndex: 0,
        rotationDeg: 0,
      };

  const cv: CvHealth = offline
    ? {
        detection: "STALLED",
        tracking: "STALLED",
        analytics: "STALLED",
        inferenceMs: 0,
        model: "YOLOv5s",
        tracker: "SORT",
        detections: 0,
        activeTracks: 0,
        device: "Simulated",
      }
    : {
        detection: degraded ? "DEGRADED" : "OK",
        tracking: degraded ? "DEGRADED" : "OK",
        analytics: degraded ? "DEGRADED" : "OK",
        inferenceMs: degraded ? +(58 + noise(14)).toFixed(0) : +(27 + noise(6)).toFixed(0),
        model: "YOLOv5s",
        tracker: "SORT",
        detections: offline ? 0 : Math.min(14, Math.round(last.queue / 3) + 2),
        activeTracks: offline ? 0 : Math.min(12, Math.round(last.queue / 4)),
        device: "Simulated",
      };

  return {
    scenario,
    state,
    stateActive: !offline,
    reason: offline
      ? "Camera signal lost: queue analytics unavailable"
      : degraded
        ? `Degraded vision pipeline: readings low-confidence · ${reasonFor(state, net, occ)}`
        : reasonFor(state, net, occ),
    queue: last.queue,
    capacity: CAPACITY,
    occupancy: occ,
    congestionSec,
    entryRate: last.entryRate,
    exitRate: last.exitRate,
    netChange: net,
    trend,
    avgWait,
    longestWait,
    predictedWait,
    clearanceTime,
    serviceTimeS,
    pacingStatus,
    throughput: Math.round(last.exitRate * 60),
    action: offline ? undefined : SIM_ACTIONS[state],
    camera,
    cv,
    recording: { active: !offline, sinceSec: s.recSinceSec, sizeGb: +(s.recSinceSec / 3600 * 2.1).toFixed(1) },
    lastKnownQueue: last.queue,
    lastUpdate: last.t,
    history: [...s.history],
    events: [...s.events],
    tracks: offline ? [] : tracksFor(last.queue, degraded),
    frameTs: s.now,
  };
}

export function useTelemetry(scenario: Scenario) {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const sim = useRef<SimState | null>(null);
  const scenRef = useRef<Scenario>(scenario);

  // reseed when scenario changes
  useEffect(() => {
    scenRef.current = scenario;
    sim.current = seed(scenario);
    setSnap(build(sim.current, scenario));
  }, [scenario]);

  useEffect(() => {
    const id = setInterval(() => {
      if (!sim.current) return;
      sim.current = step(sim.current, scenRef.current);
      setSnap(build(sim.current, scenRef.current));
    }, TICK_MS);
    return () => clearInterval(id);
  }, []);

  return snap;
}

export { CAPACITY };

// ─────────────────────────────────────────────────────────────────────────
// Real API telemetry hook — polls the FastAPI backend
// ─────────────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 1000;

function convertApiSnapshot(api: SnapshotResponse): Snapshot {
  const queue = api.queue;
  const camera = api.camera;
  const cv = api.system;

  return {
    scenario: api.scenario as Scenario,
    state: queue.state,
    stateActive: queue.state_active,
    reason: queue.reason,
    queue: queue.queue,
    capacity: queue.capacity,
    occupancy: queue.occupancy,
    congestionSec: queue.congestion_sec,
    entryRate: queue.entry_rate,
    exitRate: queue.exit_rate,
    netChange: queue.net_change,
    trend: queue.trend,
    avgWait: queue.avg_wait,
    longestWait: queue.longest_wait,
    predictedWait: queue.predicted_wait,
    clearanceTime: queue.clearance_time,
    serviceTimeS: queue.service_time_s,
    pacingStatus: queue.pacing_status,
    throughput: queue.throughput,
    action: queue.action,
    camera: {
      link: camera.link,
      fps: camera.fps,
      resolution: camera.resolution,
      latencyMs: camera.latency_ms,
      packetLoss: camera.packet_loss,
      lastFrameAgeMs: camera.last_frame_age_ms,
      uptimeSec: camera.uptime_sec,
      sourceLabel: camera.source_label || "CAM-01",
      sourceIndex: camera.source_index,
      rotationDeg: camera.rotation_deg ?? 0,
    },
    cv: {
      detection: cv.detection_status as PipelineStatus,
      tracking: cv.tracking_status as PipelineStatus,
      analytics: cv.analytics_status as PipelineStatus,
      inferenceMs: cv.inference_ms,
      model: cv.model,
      tracker: cv.tracker,
      detections: cv.detections,
      activeTracks: cv.active_tracks,
      device: cv.device ?? "",
    },
    recording: {
      active: api.recording.active,
      sinceSec: api.recording.since_sec,
      sizeGb: api.recording.size_gb,
    },
    lastKnownQueue: api.last_known_queue,
    lastUpdate: api.last_update,
    history: api.history.map((h) => ({
      t: h.t,
      queue: h.queue,
      occupancy: h.occupancy,
      entryRate: h.entry_rate,
      exitRate: h.exit_rate,
      avgWait: h.avg_wait,
    })),
    events: api.events.map((e) => ({
      id: e.id,
      t: e.timestamp,
      kind: e.kind,
      severity: e.severity,
      label: e.label,
      detail: e.detail,
      technical: e.technical,
    })),
    tracks: api.tracks.map((t) => ({
      id: t.id,
      x: t.x,
      y: t.y,
      w: t.w,
      h: t.h,
      inRoi: t.in_roi,
    })),
    frameTs: api.frame_ts,
  };
}

export function createOfflineSnapshot(errorMsg = "API disconnected"): Snapshot {
  return {
    scenario: "OFFLINE",
    state: "NORMAL",
    stateActive: false,
    reason: `Backend offline: ${errorMsg}`,
    queue: 0,
    capacity: 30,
    occupancy: 0,
    congestionSec: 0,
    entryRate: 0,
    exitRate: 0,
    netChange: 0,
    trend: "STEADY",
    avgWait: 0,
    longestWait: 0,
    throughput: 0,
    camera: {
      link: "OFFLINE",
      fps: 0,
      resolution: "-",
      latencyMs: null,
      packetLoss: null,
      lastFrameAgeMs: 0,
      uptimeSec: 0,
      sourceLabel: "",
      sourceIndex: null,
      rotationDeg: 0,
    },
    cv: {
      detection: "STALLED",
      tracking: "STALLED",
      analytics: "STALLED",
      inferenceMs: 0,
      model: "YOLOv5s",
      tracker: "SORT",
      detections: 0,
      activeTracks: 0,
      device: "",
    },
    recording: { active: false, sinceSec: null, sizeGb: null },
    lastKnownQueue: 0,
    lastUpdate: Date.now(),
    history: [],
    events: [],
    tracks: [],
    frameTs: Date.now(),
  };
}

export function useRealTelemetry(pollInterval = POLL_INTERVAL_MS) {
  const [snap, setSnap] = useState<Snapshot>(() => createOfflineSnapshot("Connecting to YOLO backend..."));
  const [error, setError] = useState<string | null>(null);

  const fetchOnce = useCallback(async () => {
    try {
      const apiSnap = await getSnapshot();
      setSnap(convertApiSnapshot(apiSnap));
      setError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch telemetry";
      setError(msg);
      setSnap(createOfflineSnapshot(msg));
    }
  }, []);

  useEffect(() => {
    fetchOnce();
    const id = setInterval(fetchOnce, pollInterval);
    return () => clearInterval(id);
  }, [fetchOnce, pollInterval]);

  return { snap, error, refetch: fetchOnce };
}

// ─────────────────────────────────────────────────────────────────────────
// Camera config hook — fetches once on mount (config rarely changes)
// ─────────────────────────────────────────────────────────────────────────

export function useCameraConfig() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(() => {});
  }, [nonce]);

  /** Re-fetch after the mapping editor persists new geometry. */
  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return [config, refresh] as const;
}
