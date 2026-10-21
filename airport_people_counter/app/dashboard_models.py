"""
Pydantic models matching the dashboard's SnapshotResponse interface.

This is the wire contract between the FastAPI backend and the React dashboard.
Fields are named in snake_case per Python convention; Pydantic will serialize
to snake_case JSON which the dashboard's convertApiSnapshot expects.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class QueueTelemetry(BaseModel):
    queue: int
    capacity: int
    occupancy: float
    congestion_sec: float
    entry_rate: float
    exit_rate: float
    net_change: float
    trend: Literal["GROWING", "CLEARING", "STEADY"]
    avg_wait: float
    longest_wait: float
    throughput: int
    state: Literal["NORMAL", "BUSY", "CONGESTED", "BACKLOG"]
    state_active: bool
    reason: str
    source_state: Optional[str] = None
    #: Operational guidance for the current condition, taken from the
    #: config-overridable recommendation table. Operator-facing language.
    action: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SystemTelemetry(BaseModel):
    inference_ms: float
    model: str
    tracker: str
    detections: int
    active_tracks: int
    detection_status: Literal["OK", "DEGRADED", "STALLED"]
    tracking_status: Literal["OK", "DEGRADED", "STALLED"]
    analytics_status: Literal["OK", "DEGRADED", "STALLED"]
    #: Compute device the inference actually ran on (e.g. "mps", "cpu").
    device: str = ""

    model_config = ConfigDict(from_attributes=True)


class CameraTelemetry(BaseModel):
    link: Literal["ONLINE", "OFFLINE", "DEGRADED"]
    fps: float
    resolution: str
    #: Network latency in ms. ``None`` when the pipeline does not measure it
    #: (the camera-only prototype has no network hop to time) -- a zero
    #: would be indistinguishable from a real measurement.
    latency_ms: Optional[float] = None
    #: Packet loss in percent. ``None`` when not measured.
    packet_loss: Optional[float] = None
    last_frame_age_ms: int
    uptime_sec: int

    model_config = ConfigDict(from_attributes=True)


class RecordingTelemetry(BaseModel):
    active: bool
    #: Seconds of the current recording segment. ``None`` when there is no
    #: recording subsystem to report on.
    since_sec: Optional[int] = None
    #: Size of the recording buffer in GB. ``None`` when not measured.
    size_gb: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class Sample(BaseModel):
    t: int
    queue: int
    occupancy: float
    entry_rate: float
    exit_rate: float
    avg_wait: float

    model_config = ConfigDict(from_attributes=True)


class EventInfo(BaseModel):
    id: int
    timestamp: int
    kind: Literal["entry", "exit", "state", "congestion", "system"]
    severity: Literal["info", "warn", "critical", "good"]
    label: str
    detail: Optional[str] = None
    #: Engineering description of the same event. Shown to technicians in the
    #: diagnostics view, never on the operator surface.
    technical: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TrackInfo(BaseModel):
    id: int
    x: float
    y: float
    w: float
    h: float
    in_roi: bool
    confidence: float = 1.0
    age: int = 0
    hits: int = 0
    state: str = "active"

    model_config = ConfigDict(from_attributes=True)


class DashboardSnapshot(BaseModel):
    scenario: str
    queue: QueueTelemetry
    system: SystemTelemetry
    camera: CameraTelemetry
    recording: RecordingTelemetry
    last_known_queue: int
    last_update: int
    history: List[Sample]
    events: List[EventInfo]
    tracks: List[TrackInfo]
    frame_ts: int

    model_config = ConfigDict(from_attributes=True)


class ConfigResponse(BaseModel):
    camera_id: str
    roi_configured: bool
    line_configured: bool
    roi_polygon: Optional[List[List[int]]] = None
    counting_line: Optional[List[List[int]]] = None
    xray_side: Optional[Literal["left", "right"]] = None
    width: int
    height: int
    #: True when a homography / pixel-per-metre calibration is configured.
    calibrated: bool = False
    #: Measurement mode: "METRES" when calibrated, otherwise "PX ONLY".
    calibration: str = "PX ONLY"
    #: The wait duration the detector treats as long enough to contribute to
    #: a backlog condition. Used as the threshold marker on the waiting time
    #: chart. Configured in config/camera.yaml, not invented here.
    wait_alert_seconds: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str
    camera: bool
    api: bool
    last_frame_age_ms: int

    model_config = ConfigDict(from_attributes=True)