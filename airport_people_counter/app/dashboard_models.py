"""
Pydantic models matching the dashboard's SnapshotResponse interface.

This is the wire contract between the FastAPI backend and the React dashboard.
Fields are named in snake_case per Python convention; Pydantic will serialize
to snake_case JSON which the dashboard's convertApiSnapshot expects.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

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
    predicted_wait: Optional[float] = None
    clearance_time: Optional[float] = None
    service_time_s: Optional[float] = None
    pacing_status: Optional[str] = "STEADY"

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
    #: Camera chrome label reported by the pipeline, e.g. "CAM-01"
    #: ("" when an older pipeline does not report it).
    source_label: str = ""
    #: Live webcam index (None for file/IP/image sources or unknown).
    source_index: Optional[int] = None
    #: Clockwise rotation applied at the capture edge.
    rotation_deg: int = 0

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

    # ── Interactive mapping (config/mappings.yaml) — additive ──────
    #: Native frame shape the per-camera geometry is calibrated against,
    #: recorded by --define-roi / the mapping editor. ``None`` when the
    #: runtime has not observed a frame yet.
    source_shape_w: Optional[int] = None
    source_shape_h: Optional[int] = None
    #: True when the running analytics is driven by a validated mapping.
    mapping_active: bool = False
    #: The active mapping reference point ("bottom_center"/"centroid").
    reference_point: Optional[str] = None

    # ── Camera source / rotation (additive) ────────────────────────
    #: Clockwise rotation persisted in camera.yaml (0/90/180/270).
    rotation: Optional[int] = None
    #: Persisted camera.yaml source when numeric ("" for files/URLs).
    source_index: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str
    camera: bool
    api: bool
    last_frame_age_ms: int

    model_config = ConfigDict(from_attributes=True)


# ===== Interactive camera mapping (config/mappings.yaml) =====


class MappingLinePayload(BaseModel):
    """One virtual counting line in normalised frame coordinates."""

    #: Exactly two ``[x, y]`` endpoints, each in ``[0, 1]``.
    points: List[List[float]]
    #: What a forward crossing of this line counts as.
    direction: Literal["IN", "OUT"]


class CameraSourceInfo(BaseModel):
    """One probed webcam index in the camera-list endpoint."""

    index: int
    available: bool
    #: The running source is marked available without probing (the probe
    #: would fail against the device the pipeline already holds open).
    running: bool = False

    model_config = ConfigDict(from_attributes=True)


class CameraSourcesResponse(BaseModel):
    sources: List[CameraSourceInfo]
    #: Persisted camera.yaml source (None when unset or non-numeric).
    current: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class CameraSelectRequest(BaseModel):
    """Body of ``POST /api/dashboard/camera/select``."""

    index: int

    model_config = ConfigDict(from_attributes=True)


class CameraRotationRequest(BaseModel):
    """Body of ``POST /api/dashboard/camera/rotation``."""

    degrees: int

    model_config = ConfigDict(from_attributes=True)


class CameraControlResult(BaseModel):
    """Result of camera select / rotation requests.

    ``applied`` is ``False`` when validation failed; ``errors`` then
    carries one human-readable reason per problem and the running
    pipeline (and the persisted config) is left untouched.
    """

    applied: bool
    camera_id: str
    source_index: Optional[int] = None
    rotation_deg: int = 0
    errors: List[str] = []

    model_config = ConfigDict(from_attributes=True)


class MappingPayload(BaseModel):
    """A per-camera mapping in normalised frame coordinates.

    Accepts exactly what the mapping editor draws: the queue ROI polygon
    and the separate IN / OUT lines. ``camera_id`` comes from the path,
    so the payload stays camera-agnostic and cannot cross-wire cameras.
    ``lines`` accepts both shapes the schema validates: the dict form
    ({"in": {...}, "out": {...}}) and the list form ([{id, ...}]).
    """

    schema_version: int = 1
    camera_id: Optional[str] = None
    roi: List[List[float]]
    lines: Union[dict, List[dict]]  # validated in depth by mapping.from_dict
    #: ``bottom_center`` (calibrated default) or ``centroid`` (legacy).
    reference_point: Optional[Literal["bottom_center", "centroid"]] = None
    #: Frame the coordinates were drawn against, for diagnostics.
    source_shape: Optional[dict] = None


class MappingApplyResult(BaseModel):
    """Result of ``PUT /api/dashboard/mapping``.

    ``applied`` is ``False`` when validation failed; ``errors`` then
    carries one human-readable reason per problem and the running
    pipeline keeps using its last known-good mapping.
    """

    applied: bool
    camera_id: str
    mapping: Optional[MappingPayload] = None
    errors: List[str] = []
    #: Where the previous file was backed up (``…mappings.yaml.bak``).
    backup_path: Optional[str] = None
    mapping_active: bool = False
    reference_point: Optional[str] = None