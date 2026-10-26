"""
Pydantic models for FastAPI responses.

Mirrors QueueSnapshot structure for serialization.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class ConfigStatus(BaseModel):
    roi_configured: bool
    line_configured: bool
    calibrated: bool
    classifier_enabled: bool


class CameraState(BaseModel):
    link: Literal["ONLINE", "OFFLINE", "DEGRADED"]
    fps: float
    resolution: str
    frame_age_ms: int
    #: Camera chrome label actually reported by the pipeline ("" = legacy
    #: process or unknown).
    source_label: str = ""
    #: Live webcam index (None for file/IP/image sources).
    source_index: Optional[int] = None
    #: Clockwise rotation applied at the capture edge.
    rotation_deg: int = 0
    #: Camera lifecycle state ("OK" | "RECONNECTING" | "NO_FRAMES" |
    #: "OFFLINE" | "" unknown), as reported by the capture supervisor.
    status: str = ""


class Track(BaseModel):
    id: int
    x: float
    y: float
    w: float
    h: float
    in_roi: bool


class Event(BaseModel):
    id: int
    t: int
    kind: Literal["entry", "exit", "state", "congestion", "system"]
    severity: Literal["info", "warn", "critical", "good"]
    label: str
    detail: Optional[str] = None


class QueueSnapshot(BaseModel):
    timestamp: str
    frame_id: int
    freshness: Literal["LIVE", "STALE", "OFFLINE"]
    staleness_ms: int

    fps: float
    inference_ms: float
    tracking_ms: float
    analytics_ms: float
    detection_count: int
    tracked_people: int

    people_in_queue: int
    total_in: int
    total_out: int
    queue_state: str
    occupancy_percent: Optional[float]
    entry_rate: float
    exit_rate: float
    avg_wait_s: float
    longest_wait_s: float
    congestion_duration_s: float
    spacing_px: Optional[float]
    movement: str

    config_status: ConfigStatus
    calibration: str
    camera_state: CameraState

    tracks: List[Track] = []
    events: List[Event] = []

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    camera: CameraState
    cv: dict
    recording: dict
    data: dict
