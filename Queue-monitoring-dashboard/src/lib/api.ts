export type QueueState = "NORMAL" | "BUSY" | "CONGESTED" | "BACKLOG";

export interface TrackInfo {
  id: number;
  x: number;
  y: number;
  w: number;
  h: number;
  in_roi: boolean;
  confidence: number;
  age: number;
  hits: number;
  state: string;
}

export interface EventInfo {
  id: number;
  timestamp: number;
  kind: "entry" | "exit" | "state" | "congestion" | "system";
  severity: "info" | "warn" | "critical" | "good";
  label: string;
  detail?: string;
  /** Engineering description of the event, for the diagnostics view only. */
  technical?: string;
}

export interface QueueTelemetry {
  queue: number;
  capacity: number;
  occupancy: number;
  congestion_sec: number;
  entry_rate: number;
  exit_rate: number;
  net_change: number;
  trend: "GROWING" | "CLEARING" | "STEADY";
  avg_wait: number;
  longest_wait: number;
  throughput: number;
  state: QueueState;
  state_active: boolean;
  reason: string;
  /** Operational guidance for the current condition. */
  action?: string;
}

export interface SystemTelemetry {
  inference_ms: number;
  model: string;
  tracker: string;
  detections: number;
  active_tracks: number;
  detection_status: string;
  tracking_status: string;
  analytics_status: string;
  /** Compute device the inference ran on ("mps", "cpu", ...). */
  device: string;
}

export interface CameraTelemetry {
  link: "ONLINE" | "OFFLINE" | "DEGRADED";
  fps: number;
  resolution: string;
  /** null when the pipeline does not measure it -- not a zero. */
  latency_ms: number | null;
  /** null when the pipeline does not measure it -- not a zero. */
  packet_loss: number | null;
  last_frame_age_ms: number;
  uptime_sec: number;
}

export interface RecordingTelemetry {
  active: boolean;
  /** null when there is no recording subsystem to report on. */
  since_sec: number | null;
  /** null when not measured. */
  size_gb: number | null;
}

export interface SnapshotResponse {
  scenario: string;
  queue: QueueTelemetry;
  system: SystemTelemetry;
  camera: CameraTelemetry;
  recording: RecordingTelemetry;
  last_known_queue: number;
  last_update: number;
  history: Array<{
    t: number;
    queue: number;
    occupancy: number;
    entry_rate: number;
    exit_rate: number;
    avg_wait: number;
  }>;
  events: EventInfo[];
  tracks: TrackInfo[];
  frame_ts: number;
}

export interface ConfigResponse {
  camera_id: string;
  roi_configured: boolean;
  line_configured: boolean;
  roi_polygon?: number[][];
  counting_line?: [[number, number], [number, number]];
  xray_side?: "left" | "right";
  width: number;
  height: number;
  /** True when a homography / pixel-per-metre calibration is configured. */
  calibrated?: boolean;
  /** Measurement mode: "METRES" when calibrated, otherwise "PX ONLY". */
  calibration?: string;
  /** Wait duration in seconds the detector treats as long enough to contribute
   *  to a backlog condition. Used as the waiting time threshold marker. */
  wait_alert_seconds?: number;
}

export interface HealthResponse {
  status: string;
  camera: boolean;
  api: boolean;
  last_frame_age_ms: number;
}

const API_BASE = "/api";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export async function getSnapshot(): Promise<SnapshotResponse> {
  return fetchJson<SnapshotResponse>("/dashboard/snapshot");
}

export async function getConfig(): Promise<ConfigResponse> {
  return fetchJson<ConfigResponse>("/dashboard/config");
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>("/health");
}

export function getVideoStreamUrl(): string {
  return `${API_BASE}/camera/stream`;
}

export function isOnline(snapshot: SnapshotResponse | null): boolean {
  return snapshot !== null && snapshot.camera.link === "ONLINE";
}

export function isDegraded(snapshot: SnapshotResponse | null): boolean {
  return snapshot !== null && snapshot.camera.link === "DEGRADED";
}

export function isOffline(snapshot: SnapshotResponse | null): boolean {
  return snapshot === null || snapshot.camera.link === "OFFLINE";
}

export async function shutdownApi(): Promise<void> {
  await fetch(`${API_BASE}/shutdown`, { method: "POST" });
}