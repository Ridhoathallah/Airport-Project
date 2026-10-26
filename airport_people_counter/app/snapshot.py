"""
Immutable snapshot layer for API serialization.

QueueAnalytics → DashboardModel → SnapshotBuilder → QueueSnapshot (frozen)
→ Thread-safe holder → FastAPI → React Dashboard
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

import numpy as np

from queue_prototype import (
    CameraConfig,
    QueueMetrics,
    SystemMetrics,
    DashboardModel,
)


def _format_duration(seconds: float) -> str:
    """Seconds as a spoken duration, e.g. "3 min 20 s"."""
    s = max(0, int(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min {sec} s"
    return f"{sec} s"


def _format_wait(seconds: float) -> str:
    """A wait time as a spoken duration, e.g. "4 min 32 s"."""
    return _format_duration(seconds)


@dataclass(frozen=True)
class ConfigStatus:
    roi_configured: bool
    line_configured: bool
    calibrated: bool
    classifier_enabled: bool


@dataclass(frozen=True)
class CameraState:
    link: Literal["ONLINE", "OFFLINE", "DEGRADED"]
    fps: float
    resolution: str
    frame_age_ms: int
    # ── Camera identity/orientation (additive; "" / None / 0 = unknown) ──
    #: Operator-facing chrome label, e.g. "CAM-01" or "WEBCAM".
    source_label: str = ""
    #: Which webcam index is live (None for file / IP / image sources).
    source_index: Optional[int] = None
    #: Clockwise rotation applied at the capture edge.
    rotation_deg: int = 0
    #: Camera lifecycle state from the capture supervisor:
    #: "OK" | "RECONNECTING" | "NO_FRAMES" | "OFFLINE" | "" (unknown).
    #: Lets the dashboard tell "camera lost, retrying" apart from
    #: "source opened but delivering no frames".
    status: str = ""


@dataclass(frozen=True)
class Track:
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


@dataclass(frozen=True)
class Event:
    id: int
    t: int  # epoch ms
    kind: Literal["entry", "exit", "state", "congestion", "system"]
    severity: Literal["info", "warn", "critical", "good"]
    label: str
    detail: Optional[str] = None
    #: Engineering description of the same event, for the diagnostics view.
    #: Never shown on the operator surface.
    technical: Optional[str] = None


@dataclass(frozen=True)
class QueueSnapshot:
    timestamp: str  # ISO8601
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
    #: The classifier's own vocabulary, passed through as-is (BUSY is a
    #: legacy alias the adapter normalises to CROWDED before the wire).
    queue_state: Literal[
        "EMPTY", "SPARSE", "NORMAL", "CROWDED", "CONGESTED",
        "BACKLOG", "UNCONFIGURED", "DATA_UNAVAILABLE",
    ]
    occupancy_percent: float
    entry_rate: float
    exit_rate: float
    avg_wait_s: float
    longest_wait_s: float
    congestion_duration_s: float
    spacing_px: Optional[float]
    movement: Literal["MOVING", "STAGNANT", "IDLE"]
    config_status: ConfigStatus
    calibration: Literal["METRES", "PX ONLY"]
    camera_state: CameraState
    tracks: List[Track] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    #: Compute device the inference ran on ("mps", "cpu", "cuda:0", ...).
    device: str = ""
    #: Epoch seconds of the newest frame the analytics actually saw
    #: (0 = no frame yet). The dashboard's "last update" is derived from
    #: this, so an offline pipeline never presents its own snapshot
    #: timestamp as the age of the newest queue reading.
    last_frame_unix: float = 0.0
    predicted_wait_s: float = 0.0
    clearance_time_s: float = 0.0
    service_time_s: float = 0.0
    pacing_status: str = "STEADY"


class SnapshotBuilder:
    @staticmethod
    def _compute_freshness(system: SystemMetrics, config: CameraConfig) -> tuple[str, int]:
        """Classify how fresh the newest frame is.

        The age is the *measured* time since the last captured frame
        (``SystemMetrics.last_frame_unix``). When that is not available
        the previous one-interval estimate is used, but it is never
        presented as a measurement.

        Returns ``(state, staleness_ms)`` where state is LIVE, STALE or
        OFFLINE.
        """
        if system.capture_fps == 0:
            # No frames flowing. If a frame ever arrived, report its real
            # age; only "never produced a frame" carries the sentinel.
            if system.last_frame_unix > 0:
                staleness = max(0.0, (time.time() - system.last_frame_unix) * 1000)
                return "OFFLINE", int(staleness)
            return "OFFLINE", 999999
        expected_interval_ms = 1000 / system.capture_fps
        if system.last_frame_unix > 0:
            staleness = max(0.0, (time.time() - system.last_frame_unix) * 1000)
        else:
            # No frame timestamp available: fall back to one interval.
            staleness = expected_interval_ms
        if staleness > expected_interval_ms * 3:
            return "STALE", int(staleness)
        return "LIVE", int(staleness)

    @staticmethod
    def _camera_state(system: SystemMetrics, config: CameraConfig) -> CameraState:
        if system.capture_fps == 0:
            link = "OFFLINE"
        elif system.capture_fps < 10:  # fallback threshold
            link = "DEGRADED"
        else:
            link = "ONLINE"

        # Report the frame size actually being analysed — the camera's
        # real capture or, when rotation ≠ 0, the rotated shape — instead
        # of repeating the requested config resolution.
        if system.frame_w > 0 and system.frame_h > 0:
            resolution = f"{system.frame_w}x{system.frame_h}"
        else:
            resolution = f"{config.width or '?'}x{config.height or '?'}"

        # Camera lifecycle state as reported by the capture supervisor.
        # Never fabricated: an empty/unexpected value falls back to a
        # link-derived default rather than claiming "OK".
        status = getattr(system, "camera_status", "")
        if not isinstance(status, str) or status not in ("OK", "RECONNECTING", "NO_FRAMES", "OFFLINE"):
            status = {"ONLINE": "OK", "DEGRADED": "OK"}.get(link, "OFFLINE")

        return CameraState(
            link=link,
            fps=system.capture_fps,
            resolution=resolution,
            frame_age_ms=int(1000 / system.capture_fps) if system.capture_fps > 0 else 0,
            source_label=system.source_label,
            source_index=system.source_index,
            rotation_deg=system.rotation_deg,
            status=status,
        )

    @staticmethod
    def _tracks_to_api(tracks: np.ndarray, config: CameraConfig) -> List[Track]:
        """
        Convert SORT tracker output (numpy array of shape [N, 5] with columns
        [x1, y1, x2, y2, track_id]) to API Track objects.
        """
        if tracks is None or len(tracks) == 0 or config.width <= 0 or config.height <= 0:
            return []
        
        from queue_prototype import _in_polygon
        w, h = float(config.width), float(config.height)
        out = []
        for t in tracks:
            x1, y1, x2, y2, track_id = t
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1
            in_roi = _in_polygon(cx, cy, config.roi) if (config.roi_configured and config.roi) else False
            out.append(Track(
                id=int(track_id),
                x=float(x1 / w) if w > 0 else 0.0,
                y=float(y1 / h) if h > 0 else 0.0,
                w=float(bw / w) if w > 0 else 0.0,
                h=float(bh / h) if h > 0 else 0.0,
                in_roi=in_roi,
            ))
        return out

    @staticmethod
    def _events_to_api(events: List[dict]) -> List[Event]:
        kind_map = {
            "line_crossing": "entry",
            "line_crossing_reversed": "system",
            "track_enter_roi": "entry",
            "congestion_start": "congestion",
            "congestion_end": "congestion",
            "state_change": "state",
            "track_lost": "system",
            "camera_switch": "system",
            "camera_rotation": "system",
        }
        severity_map = {
            "info": "info",
            "warn": "warn",
            "critical": "critical",
            "good": "good",
        }
        out = []
        for idx, e in enumerate(events):
            et = e.get("event_type", "")
            direction = e.get("direction", "")
            if et == "line_crossing":
                kind: Literal["entry", "exit", "state", "congestion", "system"] = "entry" if direction == "ENTER" else "exit"
            else:
                kind = kind_map.get(et, "system")  # type: ignore[assignment]
            sev: Literal["info", "warn", "critical", "good"] = severity_map.get(e.get("severity", "info"), "info")  # type: ignore[assignment]
            label, detail = SnapshotBuilder._operator_event_text(e)
            out.append(Event(
                id=idx,
                t=int(datetime.fromisoformat(e.get("event_time", "").replace("Z", "+00:00")).timestamp() * 1000)
                if e.get("event_time") else 0,
                kind=kind,
                severity=sev,
                label=label,
                detail=detail,
                technical=SnapshotBuilder._technical_event_text(e),
            ))
        return out

    #: Internal detection states, mapped to the words an operator reads.
    #: These names are engineering vocabulary and never reach the operator
    #: surface; the raw value stays available on the technical field.
    _STATE_WORDS = {
        "EMPTY": "Empty",
        "SPARSE": "Quiet",
        "NORMAL": "Normal",
        "BUSY": "Busy",
        "CROWDED": "Busy",
        "CONGESTED": "Congested",
        "BACKLOG": "Backlog",
        "UNCONFIGURED": "Setup incomplete",
        "DATA_UNAVAILABLE": "No data",
    }

    @classmethod
    def _operator_state(cls, state: str) -> str:
        return cls._STATE_WORDS.get(state, state.title())

    @classmethod
    def _operator_event_text(cls, e: dict) -> tuple[str, Optional[str]]:
        """(label, detail) for the operator surface.

        The detector names its states in implementation terms. The activity
        log is read by security staff during a shift, so it uses plain
        language: "Passenger joined the queue", "Queue status changed",
        "Normal -> Busy".
        """
        et = e.get("event_type", "")

        if et == "track_enter_roi":
            return "Passenger joined the queue", None

        if et == "line_crossing":
            direction = e.get("direction", "")
            if direction == "EXIT":
                waiting = e.get("waiting_s")
                detail = f"Waited {_format_wait(waiting)}" if isinstance(waiting, (int, float)) else None
                return "Passenger cleared screening", detail
            return "Passenger entered the queue area", None

        if et == "track_lost":
            return "Passenger left the camera view", None

        if et == "state_change":
            frm = cls._operator_state(str(e.get("from_state", "")))
            to = cls._operator_state(str(e.get("to_state", "")))
            return "Queue status changed", f"{frm} to {to}"

        if et == "congestion_start":
            return "Congestion detected", None

        if et == "congestion_end":
            duration = e.get("duration_s")
            detail = f"Congestion lasted {_format_duration(duration)}" if isinstance(duration, (int, float)) else None
            return "Queue beginning to clear", detail

        if et == "camera_switch":
            detail = str(e["detail"]) if e.get("detail") else None
            return "Camera source changed", detail

        if et == "camera_rotation":
            detail = str(e["detail"]) if e.get("detail") else None
            return "Image rotation changed", detail

        # Anything unexpected still reaches the operator in plain words.
        return str(e.get("label") or et.replace("_", " ").capitalize()), e.get("detail")

    @staticmethod
    def _technical_event_text(e: dict) -> str:
        """The engineering description of an event, for the diagnostics view."""
        parts = [str(e.get("event_type", "event"))]
        for key in ("track_id", "direction", "from_state", "to_state", "state",
                    "confidence", "source", "rotation"):
            if key in e and e[key] is not None:
                parts.append(f"{key}={e[key]}")
        if isinstance(e.get("waiting_s"), (int, float)):
            parts.append(f"waiting_s={e['waiting_s']}")
        if isinstance(e.get("duration_s"), (int, float)):
            parts.append(f"duration_s={e['duration_s']}")
        return " ".join(parts)

    @classmethod
    def build(
        cls,
        dashboard: DashboardModel,
        system: SystemMetrics,
        camera_config: CameraConfig,
        tracks: np.ndarray,
        recent_events: List[dict],
        annotated_frame: Optional[np.ndarray] = None,
    ) -> tuple[QueueSnapshot, Optional[np.ndarray]]:
        q = dashboard.queue
        s = dashboard.system

        freshness, staleness = cls._compute_freshness(s, camera_config)

        snap = QueueSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            frame_id=s.frame_count,
            freshness=freshness,  # type: ignore[arg-type]
            staleness_ms=staleness,
            fps=s.fps,
            inference_ms=s.inference_ms,
            tracking_ms=s.tracking_ms,
            analytics_ms=s.analytics_ms,
            detection_count=s.detections,
            tracked_people=s.tracks,
            people_in_queue=q.people_in_queue,
            total_in=q.total_entries,
            total_out=q.total_exits,
            device=s.device,
            queue_state=q.state,  # type: ignore[arg-type]
            occupancy_percent=q.occupancy_percent or 0.0,
            entry_rate=q.entry_flow,
            exit_rate=q.exit_flow,
            avg_wait_s=q.avg_wait_s,
            longest_wait_s=q.longest_wait_s,
            predicted_wait_s=getattr(q, "predicted_wait_s", 0.0),
            clearance_time_s=getattr(q, "clearance_time_s", 0.0),
            service_time_s=getattr(q, "service_time_s", 0.0),
            pacing_status=getattr(q, "pacing_status", "STEADY"),
            congestion_duration_s=q.congestion_duration_s,
            spacing_px=q.spacing_px,
            movement=q.movement,  # type: ignore[arg-type]
            config_status=ConfigStatus(
                roi_configured=camera_config.roi_configured,
                line_configured=camera_config.line_configured,
                calibrated=camera_config.calibrated,
                classifier_enabled=camera_config.roi_configured,
            ),
            calibration="METRES" if camera_config.calibrated else "PX ONLY",
            camera_state=cls._camera_state(s, camera_config),
            tracks=cls._tracks_to_api(tracks, camera_config),
            events=cls._events_to_api(recent_events),
            last_frame_unix=s.last_frame_unix,
        )
        return snap, annotated_frame

    @classmethod
    def build_offline(
        cls,
        camera_config: CameraConfig,
        system: SystemMetrics,
        recent_events: Optional[List[dict]] = None,
    ) -> QueueSnapshot:
        """Snapshot for a pipeline that has no live queue reading.

        Published by the detection loop while the camera is unavailable
        (startup failure, reconnecting, retries exhausted) so the API can
        report the camera state honestly. Queue metrics are zero-valued
        *unavailable* markers: the dashboard adapter turns them into
        explicit nulls, never into a fabricated queue reading. The camera
        block carries the real lifecycle state and the age of the last
        frame that did arrive.
        """
        last_frame_unix = float(getattr(system, "last_frame_unix", 0.0) or 0.0)
        now = time.time()
        if last_frame_unix > 0:
            staleness = max(0, int((now - last_frame_unix) * 1000))
        else:
            staleness = 999999

        snap = QueueSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            frame_id=int(system.frame_count),
            freshness="OFFLINE",
            staleness_ms=staleness,
            fps=0.0,
            inference_ms=0.0,
            tracking_ms=0.0,
            analytics_ms=0.0,
            detection_count=0,
            tracked_people=0,
            people_in_queue=0,
            total_in=0,
            total_out=0,
            device=system.device,
            queue_state="DATA_UNAVAILABLE",
            occupancy_percent=0.0,
            entry_rate=0.0,
            exit_rate=0.0,
            avg_wait_s=0.0,
            longest_wait_s=0.0,
            congestion_duration_s=0.0,
            spacing_px=None,
            movement="IDLE",
            config_status=ConfigStatus(
                roi_configured=camera_config.roi_configured,
                line_configured=camera_config.line_configured,
                calibrated=camera_config.calibrated,
                classifier_enabled=camera_config.roi_configured,
            ),
            calibration="METRES" if camera_config.calibrated else "PX ONLY",
            camera_state=cls._camera_state(system, camera_config),
            tracks=[],
            events=cls._events_to_api(recent_events or []),
            last_frame_unix=last_frame_unix,
        )
        return snap