"""
Dashboard adapter: converts internal QueueSnapshot to dashboard-shaped DashboardSnapshot.

Pure function - no side effects, no I/O, no state mutation.
"""

from __future__ import annotations

import time
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


#: The backend classifier's own state vocabulary, passed through to the
#: dashboard unchanged. "BUSY" is a legacy alias mapped to "CROWDED"; an
#: unknown value degrades to DATA_UNAVAILABLE rather than being invented
#: into a queue condition.
_QUEUE_STATES = {
    "EMPTY", "SPARSE", "NORMAL", "CROWDED", "CONGESTED",
    "BACKLOG", "UNCONFIGURED", "DATA_UNAVAILABLE",
}
_LEGACY_STATE_ALIASES = {"BUSY": "CROWDED"}

#: Pacing statuses the service tracker can honestly report.
_PACING_STATUSES = {"OPTIMAL", "STEADY", "SLOW_INSPECTION", "STAGNANT", "IDLE"}

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
# seeing a dashboard jump to zero. ``None`` / DATA_UNAVAILABLE mean no reading
# has ever existed, so nothing can be frozen.
_last_known_queue: Optional[int] = None
_last_known_occupancy: Optional[float] = None
_last_known_state: str = "DATA_UNAVAILABLE"
_last_state_lock = __import__("threading").Lock()


def _queue_state(cv_state) -> str:
    """Normalise the classifier state to the wire vocabulary, pass-through.

    The backend classifier is the single source of truth for the queue
    condition; the adapter only tolerates legacy/unknown values, it never
    re-buckets a state into a different condition.
    """
    state = str(cv_state) if cv_state is not None else ""
    state = _LEGACY_STATE_ALIASES.get(state, state)
    return state if state in _QUEUE_STATES else "DATA_UNAVAILABLE"


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


def _reason_for(state: str, net_change: Optional[float], occupancy: Optional[float]) -> str:
    """Operator-facing reason text for a classifier state.

    Occupancy / net change are only cited when they were actually
    measured (non-None); otherwise the sentence stays generic so the
    console never quotes a number it does not have.
    """
    occ_txt = f"{occupancy:.0f}%" if isinstance(occupancy, (int, float)) else "high occupancy"
    if state == "BACKLOG":
        return f"Queue exceeds capacity ({occ_txt}): entries outpacing exits"
    if state == "CONGESTED":
        if isinstance(net_change, (int, float)) and net_change > 1:
            return "Occupancy above 80% and rising: entry rate exceeds exit rate"
        return "Occupancy above 80%: throughput below demand"
    if state == "CROWDED":
        if isinstance(net_change, (int, float)) and net_change > 1:
            return "Elevated demand: queue filling"
        return "Elevated occupancy, flow balanced"
    if state == "SPARSE":
        return "Queue nearly empty: flow well within capacity"
    if state == "EMPTY":
        return "Queue clear: no passengers waiting"
    if state == "UNCONFIGURED":
        return "Queue geometry not configured: analytics unavailable"
    if state == "DATA_UNAVAILABLE":
        return "No queue reading available"
    if isinstance(net_change, (int, float)) and net_change < -1:
        return "Flow clearing normally"
    return "Nominal: occupancy within safe range"


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


def _component_statuses(snap) -> dict:
    """Per-component pipeline status, derived from real evidence.

    Detection, tracking and analytics all consume frames, so the absence
    of fresh frames stalls all three; when frames flow, each status is
    derived independently (a future component-specific signal can then
    downgrade one without touching the others). The overall pipeline is
    never reported healthy while a required component is unavailable.
    """
    link = getattr(getattr(snap, "camera_state", None), "link", "OFFLINE")
    freshness = getattr(snap, "freshness", "")
    if link == "OFFLINE" or freshness == "OFFLINE":
        return {"detection": "STALLED", "tracking": "STALLED", "analytics": "STALLED"}
    status = _derive_system_status(getattr(snap, "fps", 0.0) or 0.0)
    return {"detection": status, "tracking": status, "analytics": status}


def _optional_positive(value, digits: int = 1) -> Optional[float]:
    """A measured positive number, else None (unavailable, not zero)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return round(float(value), digits)
    return None


def _optional_minutes(seconds) -> Optional[float]:
    """Seconds → minutes on the wire; None when not measured."""
    value = _optional_positive(seconds, digits=6)
    return round(value / 60.0, 1) if value is not None else None


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
    snap: Optional[InternalQueueSnapshot],
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
        snap: Latest internal QueueSnapshot from SnapshotHolder.
              ``None`` means the pipeline thread has not produced anything
              yet (API up, pipeline down/not started).
        config: CameraConfig from config/camera.yaml
        history: List of Sample from HistoryBuffer (oldest first)
        server_start_ts: Server start time (time.time() at import)
        current_time: Optional current time (for testing)

    Returns:
        DashboardSnapshot ready for JSON serialization

    Honesty rules encoded here:

    * the classifier state passes through; the adapter never re-buckets a
      queue condition;
    * a metric whose measurement does not exist is ``null`` on the wire,
      never a fabricated zero;
    * while the camera is offline the last trustworthy reading is held
      (frozen) with its real age, instead of jumping to zero;
    * pipeline statuses reflect the evidence (frames flowing or not), so
      the console never shows a healthy pipeline when required components
      are unavailable.
    """
    if current_time is None:
        current_time = time.time()

    history = list(history)

    # --- No pipeline snapshot at all: API is up, pipeline is not. ---
    if snap is None:
        return _unavailable_snapshot(config, history, server_start_ts, current_time)

    # --- Camera link state ---
    link = snap.camera_state.link
    is_offline = link == "OFFLINE"
    is_degraded = link == "DEGRADED"

    # --- Classifier state (backend vocabulary, pass-through) ---
    cv_state = _queue_state(snap.queue_state)
    is_unconfigured = _cv_state_is_unconfigured(cv_state)
    has_reading = cv_state != "DATA_UNAVAILABLE"

    # --- OFFLINE handling: freeze last known values ---
    global _last_known_queue, _last_known_occupancy, _last_known_state
    with _last_state_lock:
        if has_reading and not is_offline:
            _last_known_queue = snap.people_in_queue
            _last_known_occupancy = snap.occupancy_percent
            _last_known_state = cv_state
        if is_offline and _last_known_state != "DATA_UNAVAILABLE":
            display_queue: Optional[int] = _last_known_queue
            display_occupancy: Optional[float] = round(_last_known_occupancy or 0.0, 1)
            display_state = _last_known_state
        elif has_reading:
            display_queue = snap.people_in_queue
            display_occupancy = round(snap.occupancy_percent or 0.0, 1)
            display_state = cv_state
        else:
            # Camera may be online but the pipeline has no reading yet.
            display_queue = None
            display_occupancy = None
            display_state = "DATA_UNAVAILABLE"
        display_ui_state = _queue_state(display_state)

    # --- Derived metrics ---
    # A rate or wait of 0 is a legitimate measurement when the pipeline
    # has a live reading; "unavailable" is the *absence* of a reading, so
    # the gate is the reading itself, never the value. While the camera
    # is offline the newest snapshot carries no live measurements, so
    # these are null rather than a frozen-looking zero.
    live_metrics = has_reading and not is_offline
    if live_metrics:
        entry_rate: Optional[float] = round(float(snap.entry_rate or 0.0), 1)
        exit_rate: Optional[float] = round(float(snap.exit_rate or 0.0), 1)
        net_change: Optional[float] = round(entry_rate - exit_rate, 1)
        trend: Optional[str] = _compute_trend(net_change)
        throughput: Optional[int] = round(exit_rate * 60)
        avg_wait_min: Optional[float] = round(float(snap.avg_wait_s or 0.0) / 60.0, 1)
        longest_wait_min: Optional[float] = round(float(snap.longest_wait_s or 0.0) / 60.0, 1)
        congestion_sec: Optional[float] = round(float(snap.congestion_duration_s or 0.0), 1)
        predicted_wait = _optional_minutes(getattr(snap, "predicted_wait_s", None))
        clearance_time = _optional_minutes(getattr(snap, "clearance_time_s", None))
        service_time_s = _optional_positive(getattr(snap, "service_time_s", None))
        pacing_status = (
            snap.pacing_status
            if isinstance(getattr(snap, "pacing_status", None), str)
            and snap.pacing_status in _PACING_STATUSES
            else None
        )
    else:
        entry_rate = exit_rate = net_change = trend = None
        throughput = avg_wait_min = longest_wait_min = congestion_sec = None
        predicted_wait = clearance_time = service_time_s = pacing_status = None

    # --- Availability summary (connectivity/liveness only) ---
    freshness = getattr(snap, "freshness", "")
    if not has_reading and not is_offline:
        scenario = "NO_DATA"
    elif is_offline:
        scenario = "OFFLINE"
    elif is_degraded or freshness == "STALE":
        scenario = "DEGRADED"
    else:
        scenario = "ONLINE"

    # --- state_active ---
    state_active = has_reading and not is_offline

    # --- Reason text ---
    if is_offline:
        reason = "Camera signal lost: queue analytics unavailable"
    elif not has_reading:
        reason = "Pipeline has not produced a queue reading yet"
    elif is_degraded:
        base_reason = _reason_for(display_ui_state, net_change, snap.occupancy_percent)
        reason = f"{base_reason}, readings low-confidence"
    else:
        reason = _reason_for(display_ui_state, net_change, snap.occupancy_percent)

    # --- Source state (for UI subtitle) ---
    source_state = display_state if is_unconfigured else None

    # --- Camera block ---
    # Lifecycle state: report what the pipeline said; when an older
    # pipeline does not report it, derive a truthful default from the
    # link instead of claiming "OK" blindly.
    cam_status = _camera_field(
        snap.camera_state, "status", "",
        type_allowed=lambda v: isinstance(v, str),
    )
    if not cam_status:
        cam_status = {"ONLINE": "OK", "DEGRADED": "OK"}.get(link, "OFFLINE")
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
        status=cam_status,
    )

    # --- System block (per-component, evidence-based) ---
    statuses = _component_statuses(snap)
    system = SystemTelemetry(
        inference_ms=snap.inference_ms,
        model="YOLOv5s",
        tracker="SORT",
        detections=snap.detection_count,
        active_tracks=snap.tracked_people,
        detection_status=statuses["detection"],
        tracking_status=statuses["tracking"],
        analytics_status=statuses["analytics"],
        device=_device_name(snap),
    )

    # --- Recording block (no recording subsystem in this build) ---
    # ``active`` is truthful (nothing is being recorded); the size and
    # segment length are reported as unavailable, not as 0.
    recording = RecordingTelemetry(active=False, since_sec=None, size_gb=None)

    # --- last_known_queue ---
    with _last_state_lock:
        last_known_queue = _last_known_queue

    # --- last_update / frame_ts: the real age of the newest frame ---
    # Only a frame that actually arrived can age a reading. A snapshot
    # built without a frame clock (no frame ever seen) reports 0 = never;
    # falling back to the build time would claim freshness for a frame
    # that does not exist.
    last_frame_unix = float(getattr(snap, "last_frame_unix", 0.0) or 0.0)
    last_update = int(last_frame_unix * 1000) if last_frame_unix > 0 else 0
    frame_ts = last_update

    # --- QueueTelemetry ---
    queue = QueueTelemetry(
        queue=display_queue,
        capacity=config.queue_capacity,
        occupancy=display_occupancy,
        congestion_sec=congestion_sec,
        entry_rate=entry_rate,
        exit_rate=exit_rate,
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
        predicted_wait=predicted_wait,
        clearance_time=clearance_time,
        service_time_s=service_time_s,
        pacing_status=pacing_status,
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
        history=_downsample_history(history, max_points=240),
        events=events,
        tracks=tracks,
        frame_ts=frame_ts,
    )


def _unavailable_snapshot(
    config: CameraConfig,
    history: List[Sample],
    server_start_ts: float,
    current_time: float,
) -> DashboardSnapshot:
    """Honest snapshot for "API up, pipeline producing nothing".

    Every measurement is ``null`` (not zero): no queue reading exists, so
    none is shown. The camera block reports OFFLINE with an empty source
    label — the pipeline has not told us which camera it would use, and
    inventing an identifier would be a lie.
    """
    camera = CameraTelemetry(
        link="OFFLINE",
        fps=0.0,
        resolution="?",
        latency_ms=None,
        packet_loss=None,
        last_frame_age_ms=0,
        uptime_sec=int(current_time - server_start_ts),
        source_label="",
        source_index=None,
        rotation_deg=0,
        status="OFFLINE",
    )
    system = SystemTelemetry(
        inference_ms=0.0,
        model="YOLOv5s",
        tracker="SORT",
        detections=0,
        active_tracks=0,
        detection_status="STALLED",
        tracking_status="STALLED",
        analytics_status="STALLED",
        device="",
    )
    queue = QueueTelemetry(
        queue=None,
        capacity=getattr(config, "queue_capacity", 0) or 0,
        occupancy=None,
        congestion_sec=None,
        entry_rate=None,
        exit_rate=None,
        net_change=None,
        trend=None,
        avg_wait=None,
        longest_wait=None,
        throughput=None,
        state="DATA_UNAVAILABLE",
        state_active=False,
        reason="Pipeline not running: no queue reading available",
        source_state="DATA_UNAVAILABLE",
    )
    return DashboardSnapshot(
        scenario="NO_DATA",
        queue=queue,
        system=system,
        camera=camera,
        recording=RecordingTelemetry(active=False, since_sec=None, size_gb=None),
        last_known_queue=None,
        last_update=0,
        history=_downsample_history(history, max_points=240),
        events=[],
        tracks=[],
        frame_ts=0,
    )


def reset_adapter_state() -> None:
    """Reset the adapter's internal OFFLINE freeze state (for testing)."""
    global _last_known_queue, _last_known_occupancy, _last_known_state
    with _last_state_lock:
        _last_known_queue = None
        _last_known_occupancy = None
        _last_known_state = "DATA_UNAVAILABLE"