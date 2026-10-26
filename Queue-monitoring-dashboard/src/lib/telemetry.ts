import { useEffect, useRef, useState, useCallback } from "react";
import { getSnapshot, getConfig, type SnapshotResponse, type ConfigResponse } from "./api";

// ─────────────────────────────────────────────────────────────────────────
// Camera-only queue telemetry. The backend is the single source of truth:
// the classifier state passes through unchanged, and a metric whose
// measurement does not exist arrives as null and stays null here. This
// layer never reclassifies a queue condition and never invents a number.
// ─────────────────────────────────────────────────────────────────────────

/** The backend classifier's own vocabulary, passed through as-is. */
export type QueueState =
  | "EMPTY"
  | "SPARSE"
  | "NORMAL"
  | "CROWDED"
  | "CONGESTED"
  | "BACKLOG"
  | "UNCONFIGURED"
  | "DATA_UNAVAILABLE";

/**
 * Connectivity, handled separately from queue classification.
 *
 * These are frontend-facing *link* states only: where the telemetry comes
 * from and whether it is flowing. The queue condition itself is always
 * `Snapshot.state` (the backend classifier's answer).
 */
export type Connection =
  | "LIVE" // backend reachable, frames flowing fresh
  | "DEGRADED" // reachable, but slow/stale low-confidence readings
  | "RECONNECTING" // reachable, camera lost, bounded retries in progress
  | "NO_FRAMES" // reachable, source opened but delivering no frames
  | "CAMERA_OFFLINE" // reachable, camera unavailable (retries stopped)
  | "NO_DATA" // reachable, pipeline has not produced a reading
  | "BACKEND_UNAVAILABLE"; // cannot reach the backend at all

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
  /**
   * Camera lifecycle state from the capture supervisor:
   * "OK" | "RECONNECTING" | "NO_FRAMES" | "OFFLINE" | "" (not reported).
   */
  status: string;
  fps: number;
  resolution: string;
  latencyMs: number | null; // null = not measured by this pipeline
  packetLoss: number | null; // %  null = not measured by this pipeline
  lastFrameAgeMs: number;
  uptimeSec: number;
  /** Camera chrome label reported by the pipeline ("" = not reported). */
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
 * through.
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
  /** Where telemetry comes from / whether it flows — never a queue state. */
  connection: Connection;
  state: QueueState;
  /** True only while a live camera reading backs the state. */
  stateActive: boolean;
  reason: string;
  /** null when no measurement exists — shown as "—", never as a fake 0. */
  queue: number | null;
  capacity: number;
  occupancy: number | null;
  congestionSec: number | null;
  entryRate: number | null;
  exitRate: number | null;
  netChange: number | null; // per min
  trend: "GROWING" | "CLEARING" | "STEADY" | null;
  avgWait: number | null; // min
  longestWait: number | null; // min
  throughput: number | null; // per hr
  /** Operational guidance for the current condition, in operator language. */
  action?: string | null;
  predictedWait?: number | null; // min (Little's Law queue wait estimation)
  clearanceTime?: number | null; // min (estimated queue clearance time)
  serviceTimeS?: number | null; // seconds per passenger
  /** null when the pipeline did not report one — never a default. */
  pacingStatus?: string | null;
  camera: CameraHealth;
  cv: CvHealth;
  recording: { active: boolean; sinceSec: number | null; sizeGb: number | null };
  /** Last trustworthy queue reading; null when none ever existed. */
  lastKnownQueue: number | null;
  lastUpdate: number; // epoch ms of last live sample (0 = never)
  history: Sample[];
  events: QueueEvent[];
  tracks: Track[];
  frameTs: number;
}

// ─────────────────────────────────────────────────────────────────────────
// Real API telemetry hook — polls the FastAPI backend
// ─────────────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 1000;

/** Derive the frontend connectivity from the backend's availability fields. */
export function deriveConnection(api: SnapshotResponse): Connection {
  // The backend knows best what the capture supervisor is doing.
  const status = api.camera.status;
  if (status === "RECONNECTING") return "RECONNECTING";
  if (status === "NO_FRAMES") return "NO_FRAMES";
  if (status === "OFFLINE" || api.camera.link === "OFFLINE" || api.scenario === "OFFLINE")
    return "CAMERA_OFFLINE";
  if (api.scenario === "NO_DATA") return "NO_DATA";
  if (api.scenario === "DEGRADED" || api.camera.link === "DEGRADED") return "DEGRADED";
  return "LIVE";
}

export function convertApiSnapshot(api: SnapshotResponse): Snapshot {
  const queue = api.queue;
  const camera = api.camera;
  const cv = api.system;

  return {
    connection: deriveConnection(api),
    state: queue.state,
    stateActive: queue.state_active,
    reason: queue.reason,
    queue: queue.queue ?? null,
    capacity: queue.capacity,
    occupancy: queue.occupancy ?? null,
    congestionSec: queue.congestion_sec ?? null,
    entryRate: queue.entry_rate ?? null,
    exitRate: queue.exit_rate ?? null,
    netChange: queue.net_change ?? null,
    trend: queue.trend ?? null,
    avgWait: queue.avg_wait ?? null,
    longestWait: queue.longest_wait ?? null,
    predictedWait: queue.predicted_wait ?? null,
    clearanceTime: queue.clearance_time ?? null,
    serviceTimeS: queue.service_time_s ?? null,
    pacingStatus: queue.pacing_status ?? null,
    throughput: queue.throughput ?? null,
    action: queue.action ?? null,
    camera: {
      link: camera.link,
      status: camera.status ?? "",
      fps: camera.fps,
      resolution: camera.resolution,
      latencyMs: camera.latency_ms,
      packetLoss: camera.packet_loss,
      lastFrameAgeMs: camera.last_frame_age_ms,
      uptimeSec: camera.uptime_sec,
      // No fabricated identifier: when the pipeline has not reported one,
      // the console shows a neutral placeholder instead of "CAM-01".
      sourceLabel: camera.source_label || "",
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
    lastKnownQueue: api.last_known_queue ?? null,
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

/**
 * An explicit "no data" snapshot: every measurement is null, the state is
 * DATA_UNAVAILABLE, and connectivity says which link is missing. Nothing
 * here is a fabricated value.
 */
export function createUnavailableSnapshot(connection: Connection, reason: string): Snapshot {
  return {
    connection,
    state: "DATA_UNAVAILABLE",
    stateActive: false,
    reason,
    queue: null,
    capacity: 30, // config default; only used for scale once readings exist
    occupancy: null,
    congestionSec: null,
    entryRate: null,
    exitRate: null,
    netChange: null,
    trend: null,
    avgWait: null,
    longestWait: null,
    throughput: null,
    camera: {
      link: "OFFLINE",
      status: connection === "RECONNECTING" || connection === "NO_FRAMES" ? connection : "OFFLINE",
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
    lastKnownQueue: null,
    lastUpdate: 0,
    history: [],
    events: [],
    tracks: [],
    frameTs: 0,
  };
}

export function useRealTelemetry(pollInterval = POLL_INTERVAL_MS) {
  const [snap, setSnap] = useState<Snapshot>(() =>
    createUnavailableSnapshot("NO_DATA", "Connecting to backend..."),
  );
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  // Last successful reading, so a poll blip marks the console unreachable
  // instead of blanking it (and vice versa: recovery needs no reload).
  const lastGood = useRef<Snapshot | null>(null);

  const fetchOnce = useCallback(async () => {
    try {
      const apiSnap = await getSnapshot();
      const converted = convertApiSnapshot(apiSnap);
      lastGood.current = converted;
      setSnap(converted);
      setError(null);
      setConnected(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch telemetry";
      setError(msg);
      setConnected(false);
      setSnap((prev) => {
        const base = lastGood.current ?? prev;
        return {
          ...base,
          connection: "BACKEND_UNAVAILABLE",
          stateActive: false,
          reason: `Backend unreachable: ${msg}`,
        };
      });
    }
  }, []);

  useEffect(() => {
    fetchOnce();
    const id = setInterval(fetchOnce, pollInterval);
    return () => clearInterval(id);
  }, [fetchOnce, pollInterval]);

  /** Manual retry: re-poll now (also used by the header Retry control). */
  return { snap, error, connected, refetch: fetchOnce };
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
