"""
Dashboard adapter: converts internal QueueSnapshot to dashboard-shaped DashboardSnapshot.

Pure function - no side effects, no I/O, no state mutation.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional, Literal

from dashboard_models import (
    DashboardSnapshot,
    QueueTelemetry,
    SystemTelemetry,
    CameraTelemetry,
    RecordingTelemetry,
    Sample,
    EventInfo,
    TrackInfo,
)

# Import internal types for type hints
from typing import TYPE_CHECKING

from snapshot_buffer import downsample as _downsample_history

if TYPE_CHECKING:
    from snapshot import QueueSnapshot as InternalQueueSnapshot
    from queue_prototype import CameraConfig
else:
    # For runtime when not in the full environment
    InternalQueueSnapshot = object
    CameraConfig = object


# CV state to UI state mapping (per spec §7)
_CV_TO_UI_STATE: dict[str, Literal["NORMAL", "BUSY", "CONGESTED", "BACKLOG"]] = {
    "EMPTY": "NORMAL",
    "SPARSE": "NORMAL",
    "NORMAL": "NORMAL",
    "CROWDED": "BUSY",
    "CONGESTED": "CONGESTED",
    "BACKLOG": "BACKLOG",
    "UNCONFIGURED": "NORMAL",
    "DATA_UNAVAILABLE": "NORMAL",
}

# CV states that indicate unconfigured
_UNCONFIGURED_CV_STATES = {"UNCONFIGURED", "DATA_UNAVAILABLE"}


def _camera_field(obj, name, default, type_allowed=None):
    """Read an additive camera attribute with a type guard.

    Mock snapshots in tests expose auto-generated attributes for any
    name; ``getattr`` alone would hand a MagicMock to pydantic. This
    only accepts values matching the wire contract, so older pipeline
    snapshots and mocks alike fall back to the documented default.
    """
    value = getattr(obj, name, default)
    if type_allowed is not None and not type_allowed(value):
        return default
    return value


# Module-level state for OFFLINE freeze. When the camera signal is lost the
# last trustworthy readings are held so the operator keeps context instead of
# seeing a dashboard jump to zero.
_last_known_queue: int = 0
_last_known_occupancy: float = 0.0
_last_known_state: str = "NORMAL"
_last_state_lock = __import__("threading").Lock()


def _iso_to_epoch_ms(iso_str: str) -> int:
    """Convert ISO8601 UTC string to epoch milliseconds."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


def _map_cv_state_to_ui(cv_state: str) -> Literal["NORMAL", "BUSY", "CONGESTED", "BACKLOG"]:
    """Map CV state to UI state per the approved mapping."""
    return _CV_TO_UI_STATE.get(cv_state, "NORMAL")


def _cv_state_is_unconfigured(cv_state: str) -> bool:
    """Check if CV state indicates unconfigured."""
    return cv_state in _UNCONFIGURED_CV_STATES


def _compute_trend(net_change: float) -> Literal["GROWING", "CLEARING", "STEADY"]:
    """Compute trend from net change."""
    if net_change > 1.0:
        return "GROWING"
    elif net_change < -1.0:
        return "CLEARING"
    return "STEADY"


def _reason_for(state: str, net_change: float, occupancy: float) -> str:
    """Generate operator-facing reason text (ported from simulator logic)."""
    if state == "BACKLOG":
        return f"Queue exceeds capacity ({occupancy:.0f}%): entries outpacing exits"
    if state == "CONGESTED":
        if net_change > 1:
            return f"Occupancy above 80% and rising: entry rate exceeds exit rate"
        return f"Occupancy above 80%: throughput below demand"
    if state == "BUSY":
        if net_change > 1:
            return f"Elevated demand: queue filling"
        return f"Elevated occupancy, flow balanced"
    if net_change < -1:
        return f"Flow clearing normally"
    return f"Nominal: occupancy within safe range"


def _device_name(snap) -> str:
    """The compute device the inference ran on, or "" when it is unknown.

    getattr keeps the adapter usable with duck-typed snapshots (the same
    pattern the track conversion already uses); the isinstance check keeps
    a non-string attribute from reaching the response model.
    """
    device = getattr(snap, "device", "")
    return device if isinstance(device, str) else ""


def _derive_system_status(fps: float) -> Literal["OK", "DEGRADED", "STALLED"]:
    """Derive pipeline status from FPS."""
    if fps >= 10.0:
        return "OK"
    elif fps > 0.0:
        return "DEGRADED"
    return "STALLED"


def _recommendation_for(state: str, config) -> Optional[str]:
    """Operational guidance for a state, in operator language.

    The wording lives in one table (queue_prototype.DEFAULT_RECOMMENDATIONS)
    and can be overridden per state in config/camera.yaml, so the dashboard
    never has to invent operational advice of its own. Returns None when the
    table cannot be consulted, which leaves the operator surface showing the
    reason text alone.
    """
    try:
        from queue_prototype import recommendation_for
    except ImportError:
        return None
    try:
        return recommendation_for(state, config)
    except Exception:
        return None


def _convert_tracks(internal_tracks: List) -> List[TrackInfo]:
    """Convert internal Track objects to TrackInfo."""
    result = []
    for t in internal_tracks:
        # Internal Track has: id, x, y, w, h, in_roi, confidence, age, hits, state
        result.append(TrackInfo(
            id=t.id,
            x=t.x,
            y=t.y,
            w=t.w,
            h=t.h,
            in_roi=t.in_roi,
            confidence=getattr(t, 'confidence', 1.0),
            age=getattr(t, 'age', 0),
            hits=getattr(t, 'hits', 0),
            state=getattr(t, 'state', "active"),
        ))
    return result


def _optional_text(value) -> Optional[str]:
    """Return the value when it is text, otherwise None.

    The dashboard endpoint must not fail because one event carries an
    unexpected type: a single bad field would blank the whole console.
    """
    if isinstance(value, str):
        return value
    return None


def _convert_events(internal_events: List) -> List[EventInfo]:
    """Convert internal Event objects to EventInfo."""
    result = []
    for e in internal_events:
        # Internal Event has: id, t, kind, severity, label, detail, technical
        result.append(EventInfo(
            id=e.id,
            timestamp=e.t,
            kind=e.kind,
            severity=e.severity,
            label=e.label,
            detail=_optional_text(getattr(e, "detail", None)),
            technical=_optional_text(getattr(e, "technical", None)),
        ))
    return result


def build_dashboard_snapshot(
    snap: InternalQueueSnapshot,
    config: CameraConfig,
    history: List[Sample],
    server_start_ts: float,
    current_time: Optional[float] = None,
) -> DashboardSnapshot:
    """
    Convert internal QueueSnapshot + config + history into DashboardSnapshot.

    This is the single source of truth for the API/dashboard contract.
    Pure function - no side effects.

    Args:
        snap: Latest internal QueueSnapshot from SnapshotHolder
        config: CameraConfig from config/camera.yaml
        history: List of Sample from HistoryBuffer (oldest first)
        server_start_ts: Server start time (time.time() at import)
        current_time: Optional current time (for testing)

    Returns:
        DashboardSnapshot ready for JSON serialization
    """
    if current_time is None:
        current_time = time.time()

    # --- Camera link state ---
    link = snap.camera_state.link
    is_offline = link == "OFFLINE"
    is_degraded = link == "DEGRADED"

    # --- CV state mapping ---
    cv_state = snap.queue_state
    ui_state = _map_cv_state_to_ui(cv_state)
    is_unconfigured = _cv_state_is_unconfigured(cv_state)

    # --- OFFLINE handling: freeze last known values ---
    global _last_known_queue, _last_known_occupancy, _last_known_state
    with _last_state_lock:
        if not is_offline:
            _last_known_queue = snap.people_in_queue
            _last_known_occupancy = snap.occupancy_percent
            _last_known_state = cv_state
        display_queue = _last_known_queue if is_offline else snap.people_in_queue
        display_occupancy = _last_known_occupancy if is_offline else snap.occupancy_percent
        display_state = _last_known_state if is_offline else cv_state
        display_ui_state = _map_cv_state_to_ui(display_state)

    # --- Derived metrics ---
    net_change = round(snap.entry_rate - snap.exit_rate, 1)
    trend = _compute_trend(net_change)
    throughput = round(snap.exit_rate * 60)  # people/hr
    avg_wait_min = round(snap.avg_wait_s / 60.0, 1)
    longest_wait_min = round(snap.longest_wait_s / 60.0, 1)

    # --- Scenario determination ---
    if is_offline:
        scenario = "OFFLINE"
    elif is_degraded and display_ui_state in ("CONGESTED", "BACKLOG"):
        scenario = "DEGRADED"
    else:
        scenario = display_ui_state

    # --- state_active ---
    state_active = not is_offline

    # --- Reason text ---
    if is_offline:
        reason = "Camera signal lost: queue analytics unavailable"
    elif is_degraded:
        base_reason = _reason_for(display_ui_state, net_change, snap.occupancy_percent)
        reason = f"{base_reason}, readings low-confidence"
    else:
        reason = _reason_for(display_ui_state, net_change, snap.occupancy_percent)

    # --- Source state (for UI subtitle) ---
    source_state = display_state if is_unconfigured else None

    # --- Camera block ---
    camera = CameraTelemetry(
        link=link,
        fps=round(snap.camera_state.fps),
        resolution=snap.camera_state.resolution,
        # The camera-only prototype has no network hop to time and no packet
        # counter to read, so these are reported as unavailable rather than
        # as a fabricated 0.
        latency_ms=None,
        packet_loss=None,
        last_frame_age_ms=snap.staleness_ms,
        uptime_sec=int(current_time - server_start_ts),
        source_label=_camera_field(
            snap.camera_state, "source_label", "",
            type_allowed=lambda v: isinstance(v, str),
        ),
        source_index=_camera_field(
            snap.camera_state, "source_index", None,
            type_allowed=lambda v: v is None or isinstance(v, int),
        ),
        rotation_deg=_camera_field(
            snap.camera_state, "rotation_deg", 0,
            type_allowed=lambda v: isinstance(v, int),
        ),
    )

    # --- System block ---
    cv_status = _derive_system_status(snap.fps)
    system = SystemTelemetry(
        inference_ms=snap.inference_ms,
        model="YOLOv5s",
        tracker="SORT",
        detections=snap.detection_count,
        active_tracks=snap.tracked_people,
        detection_status=cv_status,
        tracking_status=cv_status,
        analytics_status=cv_status,
        device=_device_name(snap),
    )

    # --- Recording block (no recording subsystem in this build) ---
    # ``active`` is truthful (nothing is being recorded); the size and
    # segment length are reported as unavailable, not as 0.
    recording = RecordingTelemetry(active=False, since_sec=None, size_gb=None)

    # --- last_known_queue ---
    last_known_queue = _last_known_queue

    # --- last_update / frame_ts from snapshot timestamp ---
    last_update = _iso_to_epoch_ms(snap.timestamp)
    frame_ts = last_update

    # --- QueueTelemetry ---
    queue = QueueTelemetry(
        queue=display_queue,
        capacity=config.queue_capacity,
        occupancy=round(display_occupancy, 1),
        congestion_sec=round(snap.congestion_duration_s, 1),
        entry_rate=snap.entry_rate,
        exit_rate=snap.exit_rate,
        net_change=net_change,
        trend=trend,
        avg_wait=avg_wait_min,
        longest_wait=longest_wait_min,
        throughput=throughput,
        state=display_ui_state,
        state_active=state_active,
        reason=reason,
        source_state=source_state,
        action=_recommendation_for(display_state, config) if state_active else None,
        predicted_wait=round(getattr(snap, "predicted_wait_s", 0.0) / 60.0, 1),
        clearance_time=round(getattr(snap, "clearance_time_s", 0.0) / 60.0, 1),
        service_time_s=round(getattr(snap, "service_time_s", 20.0), 1),
        pacing_status=getattr(snap, "pacing_status", "STEADY"),
    )

    # --- Tracks ---
    tracks = _convert_tracks(snap.tracks)

    # --- Events ---
    events = _convert_events(snap.events)

    return DashboardSnapshot(
        scenario=scenario,
        queue=queue,
        system=system,
        camera=camera,
        recording=recording,
        last_known_queue=last_known_queue,
        last_update=last_update,
        # The buffer keeps a minute of second-by-second resolution; the wire
        # format carries an evenly spaced view of it so a 1 Hz poll stays
        # small. The dashboard's time ranges are slices of this series.
        history=_downsample_history(list(history), max_points=240),
        events=events,
        tracks=tracks,
        frame_ts=frame_ts,
    )


def reset_adapter_state() -> None:
    """Reset the adapter's internal OFFLINE freeze state (for testing)."""
    global _last_known_queue, _last_known_occupancy, _last_known_state
    with _last_state_lock:
        _last_known_queue = 0
        _last_known_occupancy = 0.0
        _last_known_state = "NORMAL"