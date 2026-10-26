/** The backend classifier's own vocabulary (pass-through). */
export type QueueState =
  | "EMPTY"
  | "SPARSE"
  | "NORMAL"
  | "CROWDED"
  | "CONGESTED"
  | "BACKLOG"
  | "UNCONFIGURED"
  | "DATA_UNAVAILABLE";

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
  /** null when no measurement exists — never a fabricated zero. */
  queue: number | null;
  capacity: number;
  /** null when not measured. */
  occupancy: number | null;
  congestion_sec: number | null;
  entry_rate: number | null;
  exit_rate: number | null;
  net_change: number | null;
  trend: "GROWING" | "CLEARING" | "STEADY" | null;
  avg_wait: number | null;
  longest_wait: number | null;
  throughput: number | null;
  /** Classifier state, passed through unchanged by the backend. */
  state: QueueState;
  state_active: boolean;
  reason: string;
  /** Operational guidance for the current condition. */
  action?: string | null;
  /** Minutes; null when the estimator has no output. */
  predicted_wait?: number | null;
  /** Minutes; null when not computed. */
  clearance_time?: number | null;
  /** Seconds per passenger; null when not measured. */
  service_time_s?: number | null;
  /** null when the pipeline reported no pacing status. */
  pacing_status?: "OPTIMAL" | "STEADY" | "SLOW_INSPECTION" | "STAGNANT" | "IDLE" | null;
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
  /**
   * Camera lifecycle state from the capture supervisor:
   * "OK" | "RECONNECTING" | "NO_FRAMES" | "OFFLINE" | "" (not reported).
   * Lets the console tell "retries in progress" from "retries stopped"
   * and from "opened but delivering no frames".
   */
  status: string;
  fps: number;
  resolution: string;
  /** null when the pipeline does not measure it -- not a zero. */
  latency_ms: number | null;
  /** null when the pipeline does not measure it -- not a zero. */
  packet_loss: number | null;
  last_frame_age_ms: number;
  uptime_sec: number;
  /** Camera chrome label reported by the pipeline ("" = not reported). */
  source_label: string;
  /** Live webcam index; null for file/IP/image sources. */
  source_index: number | null;
  /** Clockwise rotation applied at the capture edge (degrees). */
  rotation_deg: number;
}

export interface RecordingTelemetry {
  active: boolean;
  /** null when there is no recording subsystem to report on. */
  since_sec: number | null;
  /** null when not measured. */
  size_gb: number | null;
}

export interface SnapshotResponse {
  /** Pipeline availability summary: "ONLINE" | "DEGRADED" | "OFFLINE" | "NO_DATA". */
  scenario: string;
  queue: QueueTelemetry;
  system: SystemTelemetry;
  camera: CameraTelemetry;
  recording: RecordingTelemetry;
  /** Last trustworthy queue count; null when none ever existed. */
  last_known_queue: number | null;
  /** Epoch ms of the newest frame the analytics saw (0 = never). */
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
  /** Interactive mapping (config/mappings.yaml) — additive fields. */
  source_shape_w?: number;
  source_shape_h?: number;
  mapping_active?: boolean;
  reference_point?: string;
  /** Clockwise capture rotation persisted in camera.yaml. */
  rotation?: number;
  /** Persisted camera.yaml source when numeric. */
  source_index?: number;
}

// ===== Interactive camera mapping (config/mappings.yaml) =====

/** The active mapping exactly as the analytics resolves it. */
export interface MappingResponse {
  camera_id: string;
  mapping_active: boolean;
  reference_point: string;
  /** "legacy" = derived from camera.yaml in memory; "mappings.yaml" = persisted. */
  mapping_source: string;
  mapping: {
    schema_version: number;
    camera_id: string;
    roi: number[][];
    lines: Array<{ id: string; points: number[][]; direction: "IN" | "OUT" }>;
    reference_point: string;
    source_shape: { width: number; height: number } | null;
    created_at_iso?: string | null;
    updated_at_iso?: string | null;
  } | null;
  source_shape_w?: number;
  source_shape_h?: number;
}

export interface MappingApplyResult {
  applied: boolean;
  camera_id: string;
  mapping: unknown;
  errors: string[];
  backup_path?: string | null;
  mapping_active: boolean;
  reference_point?: string | null;
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

export function getVideoStreamUrl(annotated: boolean = true): string {
  return annotated ? `${API_BASE}/camera/stream` : `${API_BASE}/camera/stream?annotated=0`;
}

// ===== Interactive camera mapping =====

/** Convert the wire mapping (dict/list shapes, plain numbers) to MappingData. */
export async function getMapping(): Promise<MappingResponse> {
  return fetchJson<MappingResponse>("/dashboard/mapping");
}

export async function putMapping(payload: MappingPayload): Promise<MappingApplyResult> {
  const res = await fetch(`${API_BASE}/dashboard/mapping`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

/** Wire shape of `PUT /api/dashboard/mapping`. */
export interface MappingPayload {
  schema_version: number;
  roi: number[][];
  lines: Array<{ id: "in" | "out"; points: number[][]; direction: "IN" | "OUT" }>;
  reference_point: "bottom_center" | "centroid";
  source_shape?: { width: number; height: number };
}

// ===== Camera source selection / image rotation =====

export interface CameraSourceInfo {
  index: number;
  available: boolean;
  /** True when this webcam is the one currently feeding the pipeline. */
  running: boolean;
}

export interface CameraSourcesResponse {
  sources: CameraSourceInfo[];
  /** Persisted camera.yaml source; null when unset or non-numeric. */
  current: number | null;
}

export interface CameraControlResult {
  applied: boolean;
  camera_id: string;
  source_index: number | null;
  rotation_deg: number;
  errors: string[];
}

export async function getCameraSources(): Promise<CameraSourcesResponse> {
  return fetchJson<CameraSourcesResponse>("/dashboard/camera/sources");
}

export async function selectCameraSource(index: number): Promise<CameraControlResult> {
  return postJson<CameraControlResult>("/dashboard/camera/select", { index });
}

export async function setCameraRotation(degrees: number): Promise<CameraControlResult> {
  return postJson<CameraControlResult>("/dashboard/camera/rotation", { degrees });
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export async function shutdownApi(): Promise<void> {
  await fetch(`${API_BASE}/shutdown`, { method: "POST" });
}