"""
FastAPI server for Queue Monitoring Dashboard integration.

Endpoints:
- GET /api/snapshot - Latest queue snapshot (1 Hz poll target)
- GET /api/health - Camera/CV pipeline health
- GET /api/config - Current camera configuration (read-only)
- GET /api/events - Recent events
- GET /api/camera/stream - MJPEG stream of annotated frames
- GET /api/dashboard/snapshot - Dashboard-shaped snapshot (1 Hz poll target)
- GET /api/dashboard/config - Dashboard configuration
- GET /api/dashboard/health - Dashboard health
- GET /api/dashboard/history - Dashboard history (240 samples)
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api_models import (
    QueueSnapshot as APIQueueSnapshot,
    HealthResponse as APIHealthResponse,
    CameraState as APICameraState,
    ConfigStatus as APIConfigStatus,
    Track as APITrack,
    Event as APIEvent,
)
from dashboard_models import (
    MappingApplyResult,
    MappingPayload,
    DashboardSnapshot,
    ConfigResponse,
    HealthResponse,
    CameraSourceInfo,
    CameraSourcesResponse,
    CameraSelectRequest,
    CameraRotationRequest,
    CameraControlResult,
)
from dashboard_adapter import build_dashboard_snapshot, reset_adapter_state
from snapshot_buffer import downsample, get_history_buffer
from snapshot_holder import get_holder
from queue_prototype import CameraConfig
from orientation import VALID_ROTATIONS
from shutdown import request_shutdown, start_watchdog


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Airport X-Ray Queue Monitoring API",
    version="1.0.0",
    lifespan=lifespan,
)

# Server start time for uptime calculation
SERVER_START_TS = time.time()

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8443", "http://127.0.0.1:8443"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _config_rotation(config) -> int:
    """Read camera rotation defensively (0 when absent/invalid)."""
    try:
        value = int(getattr(config, "rotation", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return value if value in VALID_ROTATIONS else 0


def _config_source_index(config) -> Optional[int]:
    """Read camera source defensively; numeric only when really numeric."""
    source = str(getattr(config, "source", "") or "")
    return int(source) if source.isdigit() else None


def _camera_field(obj, name, default):
    """Return an additive camera attribute when its type fits the wire
    contract, else the default.

    Mock snapshots in tests expose auto-generated attributes for any
    name, so ``getattr`` alone would hand a MagicMock to pydantic; this
    guards the additive fields without hard-depending on every mock.
    """
    value = getattr(obj, name, default)
    if isinstance(value, (str, int, float)) and value is not default:
        return value
    return default


def _convert_snap(snap) -> Optional[APIQueueSnapshot]:
    """Convert internal QueueSnapshot to API QueueSnapshot."""
    if snap is None:
        return None
    
    # Convert tracks
    api_tracks = [
        APITrack(
            id=t.id,
            x=t.x,
            y=t.y,
            w=t.w,
            h=t.h,
            in_roi=t.in_roi,
        )
        for t in snap.tracks
    ]
    
    # Convert events
    api_events = [
        APIEvent(
            id=e.id,
            t=e.t,
            kind=e.kind,
            severity=e.severity,
            label=e.label,
            detail=e.detail,
        )
        for e in snap.events
    ]
    
    # Convert config status
    api_config = APIConfigStatus(
        roi_configured=snap.config_status.roi_configured,
        line_configured=snap.config_status.line_configured,
        calibrated=snap.config_status.calibrated,
        classifier_enabled=snap.config_status.classifier_enabled,
    )
    
    # Convert camera state
    api_camera = APICameraState(
        link=snap.camera_state.link,
        fps=snap.camera_state.fps,
        resolution=snap.camera_state.resolution,
        frame_age_ms=snap.camera_state.frame_age_ms,
        source_label=_camera_field(snap.camera_state, "source_label", ""),
        source_index=_camera_field(snap.camera_state, "source_index", None),
        rotation_deg=_camera_field(snap.camera_state, "rotation_deg", 0),
    )
    
    return APIQueueSnapshot(
        timestamp=snap.timestamp,
        frame_id=snap.frame_id,
        freshness=snap.freshness,
        staleness_ms=snap.staleness_ms,
        fps=snap.fps,
        inference_ms=snap.inference_ms,
        tracking_ms=snap.tracking_ms,
        analytics_ms=snap.analytics_ms,
        detection_count=snap.detection_count,
        tracked_people=snap.tracked_people,
        people_in_queue=snap.people_in_queue,
        total_in=snap.total_in,
        total_out=snap.total_out,
        queue_state=snap.queue_state,
        occupancy_percent=snap.occupancy_percent,
        entry_rate=snap.entry_rate,
        exit_rate=snap.exit_rate,
        avg_wait_s=snap.avg_wait_s,
        longest_wait_s=snap.longest_wait_s,
        congestion_duration_s=snap.congestion_duration_s,
        spacing_px=snap.spacing_px,
        movement=snap.movement,
        config_status=api_config,
        calibration=snap.calibration,
        camera_state=api_camera,
        tracks=api_tracks,
        events=api_events,
    )


@app.get("/api/snapshot", response_model=Optional[APIQueueSnapshot])
async def get_snapshot():
    """Get the latest queue snapshot for dashboard polling."""
    holder = get_holder()
    snap = holder.get_latest_snapshot()
    return _convert_snap(snap)


@app.get("/api/health", response_model=APIHealthResponse)
async def get_health():
    """Get camera and CV pipeline health status."""
    holder = get_holder()
    snap = holder.get_latest_snapshot()

    if snap is None:
        return APIHealthResponse(
            camera=APICameraState(link="OFFLINE", fps=0.0, resolution="?", frame_age_ms=0),
            cv={"detection": "STALLED", "tracking": "STALLED", "analytics": "STALLED"},
            recording={"active": False},
            data={"status": "NO_DATA"},
        )

    cv_status = "OK" if snap.fps > 0 else "STALLED"
    api_snap = _convert_snap(snap)  # type: ignore[assignment]
    return APIHealthResponse(
        camera=api_snap.camera_state,
        cv={
            "detection": cv_status,
            "tracking": cv_status,
            "analytics": cv_status,
            "inference_ms": snap.inference_ms,
            "model": "YOLOv5s",
            "tracker": "SORT",
            "detections": snap.detection_count,
            "active_tracks": snap.tracked_people,
        },
        recording={"active": False},
        data={"status": snap.freshness, "staleness_ms": snap.staleness_ms},
    )


@app.get("/api/config")
async def get_config():
    """Get current camera configuration (read-only)."""
    from queue_prototype import load_camera_config
    from detect_people import DEFAULT_CAMERA_CONFIG_PATH
    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)

    return {
        "camera_id": config.camera_id,
        "width": config.width,
        "height": config.height,
        "roi_configured": config.roi_configured,
        "line_configured": config.line_configured,
        "xray_side": getattr(config, 'xray_side_point', None) and "right" or None,
        "roi_polygon": config.roi if config.roi_configured else None,
        "counting_line": config.counting_line if config.line_configured else None,
        "classifier": {
            "occupancy_sparse": config.occupancy_sparse,
            "occupancy_normal": config.occupancy_normal,
            "occupancy_crowded": config.occupancy_crowded,
            "occupancy_congested": config.occupancy_congested,
            "hysteresis": config.hysteresis,
            "debounce_frames": config.debounce_frames,
        },
        "backlog": {
            "min_people": config.backlog_min_people,
            "min_occupancy": config.backlog_min_occupancy,
            "min_wait_seconds": config.backlog_min_wait_seconds,
            "persistence_seconds": config.backlog_persistence_seconds,
        },
    }


@app.get("/api/events")
async def get_events(limit: int = Query(50, ge=1, le=200)):
    """Get recent events."""
    holder = get_holder()
    snap = holder.get_latest_snapshot()

    if snap is None:
        return {"events": []}

    events = snap.events[:limit]
    return {
        "events": [
            {
                "id": e.id,
                "t": e.t,
                "kind": e.kind,
                "severity": e.severity,
                "label": e.label,
                "detail": e.detail,
                "technical": getattr(e, "technical", None),
            }
            for e in events
        ]
    }


async def _mjpeg_generator(annotated: bool = True):
    """Generate MJPEG stream from frames in the snapshot holder.

    ``annotated=True`` streams the operator view (YOLO boxes, ROI,
    counting line); ``annotated=False`` streams the same raw camera
    picture the mapping editor draws on, so calibration overlays are
    rendered by the editor itself instead of being burned into the
    pixels. Falls back to the annotated frame when no raw frame is
    available (older pipeline processes).
    """
    holder = get_holder()
    boundary = "frame"

    while True:
        frame = holder.get_latest_frame() if annotated else holder.get_latest_raw_frame()
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode() + jpeg.tobytes() + b"\r\n"
        await asyncio.sleep(1 / 30)


@app.get("/api/camera/stream")
async def camera_stream(annotated: int = Query(1, ge=0, le=1)):
    """MJPEG stream of camera frames.

    ``?annotated=1`` (default) burns in detections + geometry for the
    monitoring view. ``?annotated=0`` serves the raw camera picture for
    the mapping editor. Existing consumers are unaffected: the default
    is identical to the pre-mapping behaviour.
    """
    return StreamingResponse(
        _mjpeg_generator(annotated=bool(annotated)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ===== Dashboard API Endpoints =====

@app.get("/api/dashboard/snapshot", response_model=DashboardSnapshot)
async def get_dashboard_snapshot():
    """Get the latest dashboard-shaped snapshot for dashboard polling (1 Hz)."""
    holder = get_holder()
    snap = holder.get_latest_snapshot()

    if snap is None:
        # Return a minimal snapshot with defaults
        from queue_prototype import load_camera_config
        from detect_people import DEFAULT_CAMERA_CONFIG_PATH
        config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
        history_buf = get_history_buffer()
        return build_dashboard_snapshot(
            snap=MockEmptySnapshot(),
            config=config,
            history=history_buf.snapshot(),
            server_start_ts=SERVER_START_TS,
        )

    from queue_prototype import load_camera_config
    from detect_people import DEFAULT_CAMERA_CONFIG_PATH
    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
    history_buf = get_history_buffer()

    return build_dashboard_snapshot(
        snap=snap,
        config=config,
        history=history_buf.snapshot(),
        server_start_ts=SERVER_START_TS,
    )


@app.get("/api/dashboard/config", response_model=ConfigResponse)
async def get_dashboard_config():
    """Get current camera configuration for dashboard."""
    from queue_prototype import load_camera_config
    from detect_people import DEFAULT_CAMERA_CONFIG_PATH
    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)

    # Determine xray_side
    xray_side = None
    if hasattr(config, 'xray_side_point') and config.xray_side_point is not None:
        xray_side = "right" if config.xray_side_point[0] > config.width / 2 else "left"

    return ConfigResponse(
        camera_id=config.camera_id,
        roi_configured=config.roi_configured,
        line_configured=config.line_configured,
        roi_polygon=config.roi if config.roi_configured else None,
        counting_line=config.counting_line if config.line_configured else None,
        xray_side=xray_side,
        width=config.width,
        height=config.height,
        calibrated=bool(config.calibrated),
        calibration="METRES" if config.calibrated else "PX ONLY",
        wait_alert_seconds=getattr(config, "backlog_min_wait_seconds", None),
        # ── additive mapping fields (interactive calibration) ──
        source_shape_w=getattr(config, "source_shape_w", None),
        source_shape_h=getattr(config, "source_shape_h", None),
        mapping_active=bool(getattr(config, "mapping_active", False)),
        reference_point=_safe_reference_point(config),
        rotation=_config_rotation(config),
        source_index=_config_source_index(config),
    )


@app.get("/api/dashboard/mapping")
async def get_dashboard_mapping():
    """Get the active per-camera mapping in normalised coordinates.

    Returns the mapping the analytics actually uses: the persisted
    ``config/mappings.yaml`` entry when one is valid, otherwise the
    mapping derived in-memory from the legacy ``camera.yaml`` geometry
    (``mapping_source: "legacy"``). ``mapping`` is ``null`` when the
    camera is not calibrated at all.
    """
    from queue_prototype import load_camera_config
    from detect_people import DEFAULT_CAMERA_CONFIG_PATH
    from mapping import MappingGeometry

    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
    mapping = getattr(config, "mapping", None)
    return {
        "camera_id": config.camera_id,
        "mapping_active": bool(getattr(config, "mapping_active", False)),
        "reference_point": _safe_reference_point(config) or "centroid",
        "mapping_source": getattr(config, "mapping_source", "none"),
        "mapping": mapping.to_dict() if isinstance(mapping, MappingGeometry) else None,
        "source_shape_w": getattr(config, "source_shape_w", None),
        "source_shape_h": getattr(config, "source_shape_h", None),
    }


def _safe_reference_point(config) -> Optional[str]:
    """The active reference point, tolerant of mock configs in tests."""
    mapping = getattr(config, "mapping", None)
    ref = getattr(mapping, "reference_point", None)
    if isinstance(ref, str) and ref:
        return ref
    ref = getattr(config, "reference_point", None)
    return ref if isinstance(ref, str) and ref else None


@app.put("/api/dashboard/mapping", response_model=MappingApplyResult)
async def put_dashboard_mapping(payload: MappingPayload):
    """Validate, apply and persist a camera mapping.

    Order guarantees:

    1. Validation happens BEFORE anything is written — an invalid
       payload never touches disk, so the running pipeline keeps its
       last known-good mapping (no rollback dance needed at read time).
    2. Persistence is atomic: serialise → re-parse → tempfile →
       ``os.replace``, plus a ``mappings.yaml.bak`` of the previous
       content.
    3. Only a successful write asks the detection loop to hot-reload.
    """
    import os

    from dashboard_models import MappingApplyResult, MappingPayload  # re-export convenience
    from detect_people import DEFAULT_CAMERA_CONFIG_PATH
    from mapping import (
        MappingGeometry,
        MappingError,
        save_mapping,
        default_mappings_path,
        validate_mapping,
    )
    from reload_signal import request_config_reload

    from queue_prototype import load_camera_config

    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
    camera_id = config.camera_id or "CAMERA 01"

    data = payload.model_dump()
    data["camera_id"] = camera_id
    # The frame shape for physical checks: prefer what the mapping was
    # drawn against (client-provided), else the runtime's known shape.
    if not data.get("source_shape") and config.source_shape_w and config.source_shape_h:
        data["source_shape"] = {
            "width": config.source_shape_w,
            "height": config.source_shape_h,
        }

    try:
        mapping = MappingGeometry.from_dict(data, camera_id=camera_id)
    except MappingError as exc:
        return MappingApplyResult(applied=False, camera_id=camera_id, errors=[str(exc)])

    if data.get("source_shape"):
        errors = validate_mapping(
            mapping,
            shape=(int(data["source_shape"]["width"]), int(data["source_shape"]["height"])),
        )
    else:
        errors = validate_mapping(mapping)
    if errors:
        return MappingApplyResult(applied=False, camera_id=camera_id, errors=errors)

    try:
        backup_path = save_mapping(default_mappings_path(DEFAULT_CAMERA_CONFIG_PATH), mapping)
    except (MappingError, OSError) as exc:
        return MappingApplyResult(applied=False, camera_id=camera_id, errors=[f"persist failed: {exc}"])

    # Success: the next detection-loop frame picks the new mapping up.
    request_config_reload("api")
    return MappingApplyResult(
        applied=True,
        camera_id=camera_id,
        mapping=payload,
        errors=[],
        backup_path=str(backup_path) if backup_path else None,
        mapping_active=True,
        reference_point=mapping.reference_point,
    )


# ===== Camera source selection / image rotation (additive) =====

#: Light camera-probe cache: probing opens the device briefly, so the
#: list endpoint caches results instead of hammering every poll.
_camera_probe_cache: dict = {"ts": 0.0, "available": {}}
_CAMERA_PROBE_RANGE = range(5)  # webcam indexes 0–4
_CAMERA_PROBE_TTL_S = 10.0


def _probed_sources_dict() -> dict:
    """Probe webcam indexes 0–4 (same thread as blocking callers)."""
    from detect_people import probe_camera_index

    now = time.monotonic()
    if now - _camera_probe_cache["ts"] < _CAMERA_PROBE_TTL_S and _camera_probe_cache["available"]:
        return dict(_camera_probe_cache["available"])
    available = {}
    for idx in _CAMERA_PROBE_RANGE:
        ok, _ = probe_camera_index(idx)
        available[idx] = ok
    _camera_probe_cache["ts"] = now
    _camera_probe_cache["available"] = available
    return dict(available)


@app.get("/api/dashboard/camera/sources", response_model=CameraSourcesResponse)
async def get_camera_sources():
    """List webcam sources for the selection control.

    Probes indexes 0–4 with a light ~1.5 s-max probe each, cached for
    10 s. The currently running source is reported available without a
    probe — the probe cannot open a device the pipeline already holds.
    """
    from dashboard_models import CameraSourcesResponse, CameraSourceInfo
    from camera_config_store import load_camera_yaml
    from detect_people import DEFAULT_CAMERA_CONFIG_PATH
    from queue_prototype import load_camera_config

    loop = asyncio.get_running_loop()
    probed = await loop.run_in_executor(None, _probed_sources_dict)
    holder = get_holder()
    snap = holder.get_latest_snapshot()
    running_index = getattr(getattr(snap, "camera_state", None), "source_index", None)

    doc = load_camera_yaml(DEFAULT_CAMERA_CONFIG_PATH)
    cam_src = (doc.get("camera", {}) or {}).get("source")
    persisted = int(cam_src) if isinstance(cam_src, int) else (
        int(str(cam_src)) if isinstance(cam_src, str) and str(cam_src).isdigit() else None
    )

    sources = []
    for idx in _CAMERA_PROBE_RANGE:
        is_running = running_index is not None and idx == int(running_index)
        sources.append(CameraSourceInfo(
            index=idx,
            available=bool(probed.get(idx)) or is_running,
            running=is_running,
        ))
    return CameraSourcesResponse(sources=sources, current=persisted)


@app.post("/api/dashboard/camera/select", response_model=CameraControlResult)
async def select_camera_source(payload: CameraSelectRequest):
    """Select which webcam feeds the lane — hot, no pipeline restart.

    Guarantees (same pattern as the mapping endpoint):

    1. Availability is probed BEFORE anything is written; an unavailable
       index never touches disk or the running pipeline.
    2. camera.yaml is updated atomically with a ``.bak``.
    3. Only a successful persist asks the loop to swap the capture.
       If opening the new camera fails, the loop keeps the old one and
       logs the error — the config change is the operator intent and is
       retried straight away on the next switch request.
    """
    from dashboard_models import CameraControlResult, CameraSelectRequest  # noqa: F401
    from camera_config_store import update_camera_yaml
    from camera_control import request_camera_control
    from detect_people import probe_camera_index, DEFAULT_CAMERA_CONFIG_PATH
    from queue_prototype import load_camera_config

    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
    camera_id = config.camera_id or "CAMERA 01"
    index = int(payload.index)

    if index < 0 or index not in _CAMERA_PROBE_RANGE:
        return CameraControlResult(
            applied=False, camera_id=camera_id,
            errors=[f"source index {index} is outside the supported range 0–{max(_CAMERA_PROBE_RANGE)}"],
        )

    running_index = getattr(
        getattr(get_holder().get_latest_snapshot(), "camera_state", None),
        "source_index", None,
    )
    if running_index is not None and index == int(running_index):
        return CameraControlResult(
            applied=True, camera_id=camera_id,
            source_index=index,
            rotation_deg=_config_rotation(config),
            errors=[],
        )

    loop = asyncio.get_running_loop()
    available, _ = await loop.run_in_executor(None, probe_camera_index, index)
    if not available:
        return CameraControlResult(
            applied=False, camera_id=camera_id, source_index=index,
            errors=[f"camera {index} is not available — nothing was changed"],
        )

    try:
        update_camera_yaml(DEFAULT_CAMERA_CONFIG_PATH, {"camera.source": str(index)})
    except (ValueError, OSError) as exc:
        return CameraControlResult(
            applied=False, camera_id=camera_id, source_index=index,
            errors=[f"camera.yaml update failed: {exc}"],
        )

    request_camera_control({"action": "switch", "source": str(index)}, "api")
    return CameraControlResult(
        applied=True, camera_id=camera_id, source_index=index,
        rotation_deg=_config_rotation(config), errors=[],
    )


@app.post("/api/dashboard/camera/rotation", response_model=CameraControlResult)
async def set_camera_rotation(payload: CameraRotationRequest):
    """Rotate the captured image 90°-step clockwise.

    Rotation is applied at the capture edge (before detection), and the
    active calibration geometry is rotated by the same delta so the
    calibrated region keeps covering the same physical scene area:

    - mapping-driven geometry: ``mappings.yaml`` entry is rotated and
      saved atomically (``mappings.yaml.bak``);
    - legacy pixel geometry: ``camera.yaml`` ROI/line/xray side points
      and ``source.shape_w/h`` are rotated in the same atomic write as
      the new ``camera.rotation`` value.

    A no-op rotation (same value) applies nothing. A failed geometry
    transform leaves both files untouched.
    """
    from dashboard_models import CameraControlResult, CameraRotationRequest  # noqa: F401
    from camera_config_store import update_camera_yaml
    from camera_control import request_camera_control
    from mapping import save_mapping, default_mappings_path, validate_mapping, MappingError
    from orientation import rotate_mapping, transform_legacy_layout, delta_rotation, rotated_size
    from queue_prototype import load_camera_config
    from detect_people import DEFAULT_CAMERA_CONFIG_PATH

    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
    camera_id = config.camera_id or "CAMERA 01"
    degrees = int(payload.degrees)
    current = _config_rotation(config)
    result = CameraControlResult(
        applied=True, camera_id=camera_id,
        source_index=_config_source_index(config),
        rotation_deg=degrees, errors=[],
    )
    if degrees not in VALID_ROTATIONS:
        result.applied = False
        result.errors = [f"rotation must be one of {VALID_ROTATIONS}"]
        return result
    delta = delta_rotation(current, degrees)
    if delta == 0:
        return result

    updates: dict = {"camera.rotation": degrees}
    if getattr(config, "mapping_source", "") == "mappings.yaml" and getattr(config, "mapping", None) is not None:
        # Mapping-driven lane: rotate the persisted mappings.yaml entry.
        rotated = rotate_mapping(config.mapping, delta)
        errors = validate_mapping(rotated, rotated.source_shape)
        if errors:
            result.applied = False
            result.errors = [f"rotation would break the calibration: {'; '.join(errors)}"]
            return result
        try:
            save_mapping(default_mappings_path(DEFAULT_CAMERA_CONFIG_PATH), rotated)
        except (MappingError, OSError) as exc:
            result.applied = False
            result.errors = [f"mappings.yaml update failed: {exc}"]
            return result
    elif config.roi or config.counting_line or config.xray_side_point:
        # Legacy pixel geometry: rotate the camera.yaml points in the
        # same atomic write as the rotation value. The legacy-derived
        # mapping is in-memory only and must never be persisted back
        # (see load_camera_config), so the pixels stay authoritative.
        size = (
            int(config.source_shape_w or config.width or 0),
            int(config.source_shape_h or config.height or 0),
        )
        if size[0] <= 0 or size[1] <= 0:
            result.applied = False
            result.errors = [
                "legacy pixel geometry needs a known frame shape — set camera.source.shape_w/h first",
            ]
            return result
        legacy = transform_legacy_layout(
            config.roi, config.counting_line, config.xray_side_point, delta, size,
        )
        updates.update({f"camera.{key}": value for key, value in legacy.items()})
        new_w, new_h = rotated_size(size[0], size[1], delta)
        updates["source.shape_w"] = new_w
        updates["source.shape_h"] = new_h

    try:
        update_camera_yaml(DEFAULT_CAMERA_CONFIG_PATH, updates)
    except (ValueError, OSError) as exc:
        # camera.yaml is authoritative for rotation; without this write
        # the mapping file must stay pre-rotation too.
        if getattr(config, "mapping_source", "") == "mappings.yaml" and getattr(config, "mapping", None) is not None:
            try:
                save_mapping(default_mappings_path(DEFAULT_CAMERA_CONFIG_PATH), config.mapping)
            except (MappingError, OSError):
                pass  # reported below regardless
        result.applied = False
        result.errors = [f"camera.yaml update failed: {exc}"]
        return result

    request_camera_control({"action": "rotate", "rotation": degrees}, "api")
    return result


@app.get("/api/dashboard/health", response_model=HealthResponse)
async def get_dashboard_health():
    """Get dashboard health status."""
    holder = get_holder()
    snap = holder.get_latest_snapshot()

    if snap is None:
        return HealthResponse(
            status="NO_DATA",
            camera=False,
            api=True,
            last_frame_age_ms=0,
        )

    camera_ok = snap.camera_state.link == "ONLINE"
    last_frame_age_ms = snap.staleness_ms
    status = "OK" if camera_ok else "DEGRADED"

    return HealthResponse(
        status=status,
        camera=camera_ok,
        api=True,
        last_frame_age_ms=last_frame_age_ms,
    )


@app.get("/api/dashboard/history")
async def get_dashboard_history(points: int = Query(240, ge=2, le=3600)):
    """Get dashboard history.

    The buffer holds one sample per second for the last hour. ``points``
    selects how many evenly spaced samples to return, newest last.
    """
    history_buf = get_history_buffer()
    samples = downsample(history_buf.snapshot(), max_points=points)
    return {"samples": [s.model_dump() for s in samples]}


#: The uvicorn server instance, when this module started it. Captured so
#: the shutdown endpoint can ask uvicorn to stop serving gracefully.
_server_ref = None


def _shutdown_server():
    """Request a graceful shutdown of the whole process.

    The detection loop polls the shutdown flag and exits its loop, which
    releases the camera and closes its outputs; uvicorn is asked to stop
    accepting connections. A watchdog remains as a last resort so a
    wedged process still terminates.
    """
    request_shutdown("api")
    server = _server_ref
    if server is not None:
        server.should_exit = True
    start_watchdog(timeout_s=10.0)


@app.post("/api/shutdown")
async def shutdown_server():
    """Stop the backend: the detection loop exits, then the process ends."""
    _shutdown_server()
    return {"status": "shutting_down"}


# Mock empty snapshot for when no data available
class MockEmptySnapshot:
    """Minimal snapshot-like object for empty state."""
    people_in_queue = 0
    queue_state = "DATA_UNAVAILABLE"
    occupancy_percent = 0.0
    congestion_duration_s = 0.0
    entry_rate = 0.0
    exit_rate = 0.0
    avg_wait_s = 0.0
    longest_wait_s = 0.0
    fps = 0.0
    inference_ms = 0.0
    detection_count = 0
    tracked_people = 0
    timestamp = "2024-01-01T00:00:00Z"
    tracks = []
    events = []
    staleness_ms = 0
    camera_state = type('obj', (object,), {
        'link': 'OFFLINE',
        'fps': 0.0,
        'resolution': '?',
    })()


def run_api_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the FastAPI server.

    The uvicorn instance is kept so that ``POST /api/shutdown`` can ask it
    to stop serving gracefully instead of killing the process.
    """
    import uvicorn

    global _server_ref
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    _server_ref = server
    try:
        server.run()
    finally:
        _server_ref = None