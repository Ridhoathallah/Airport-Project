"""
Camera-only queue analytics prototype.

    detect_people.py   (vision: YOLO person detection + SORT tracking)
          |
    queue_prototype.py (this module: QueueAnalytics)
          |
    snapshot + events + overlay

Everything here is camera-relative. Until a homography / px-per-metre
calibration is supplied, no real-world (metre) measurement is produced.
The four queue concepts are kept strictly separate:

  occupancy   people whose centroid is inside the queue ROI
  throughput  people crossing the processing/EXIT line per unit time
  waiting     entry -> exit duration per tracked person
  congestion  rule-based state from many variables, not just a head count

The classifier is transparent (rule-based): every snapshot carries the
exact reasons that produced its state, so the prototype can be debugged
and calibrated against real CCTV footage.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from orientation import VALID_ROTATIONS

try:  # Mapping schema lives next to this module when running inputs.
    from mapping import (  # type: ignore
        BOTTOM_CENTER_REFERENCE,
        CENTROID_REFERENCE,
        MappingGeometry,
        MappingError,
        is_simple_polygon,
        to_pixel as mapping_to_pixel,
        validate_mapping as validate_mapping_geometry,
    )
    _MAPPING_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive imports
    _MAPPING_AVAILABLE = False
    MappingGeometry = None
    MappingError = ValueError
    BOTTOM_CENTER_REFERENCE = "bottom_center"
    is_simple_polygon = None
    mapping_to_pixel = None
    validate_mapping_geometry = None
    CENTROID_REFERENCE = "centroid"

# Camera prototype states (a subset of the future sensor-fusion states).
EMPTY = "EMPTY"
SPARSE = "SPARSE"
NORMAL = "NORMAL"
CROWDED = "CROWDED"
CONGESTED = "CONGESTED"
BACKLOG = "BACKLOG"
UNCONFIGURED = "UNCONFIGURED"
SEVERE_STATES = (CONGESTED, BACKLOG)

BACKLOG_UNAVAILABLE_MSG = "BACKLOG CHECK UNAVAILABLE - FLOW BASELINE NOT CONFIGURED"
PIXEL_DENSITY_LABEL = "PIXEL DENSITY - UNCALIBRATED"

# Crossing events are suppressed for this many consecutive frames after a
# track crosses a line, so jitter right next to a line produces exactly one
# event instead of a burst. Tuned for the 15-30 fps monitoring feeds this
# prototype runs at, and deliberately separate from
# `classifier.debounce_frames` (which is the state-machine anti-flapping
# knob and must keep that meaning).
#
# Semantics: `crossing_debounce_frames` is the number of *consecutive
# frames* a track must be observed on the far side of a counting line
# before the crossing is committed and becomes an event:
#   1 -> fires on the frame the side first changes (pre-mapping behaviour)
#   2 -> one additional stable frame, which filters 1-frame jitter
#   3 -> two stable frames, for noisy tracking feeds
DEFAULT_CROSSING_DEBOUNCE_FRAMES = 2

# Status colours (BGR) — distinct per state, bright enough on dark panels.
# EMPTY stays grey/blue, SPARSE/NORMAL are green, CROWDED yellow,
# CONGESTED orange, BACKLOG red, UNCONFIGURED grey. Always shown with text.
_STATE_COLORS = {
    UNCONFIGURED: (130, 130, 130),
    EMPTY: (190, 185, 120),
    SPARSE: (0, 200, 0),
    NORMAL: (0, 230, 118),
    CROWDED: (0, 215, 215),
    CONGESTED: (0, 140, 255),
    BACKLOG: (0, 40, 255),
}

# Operational recommendations shown for each state. The operator can
# override any of these in config/camera.yaml under `recommendations:`.
DEFAULT_RECOMMENDATIONS = {
    UNCONFIGURED: "Configure the queue ROI, counting line and X-ray side point to enable analytics.",
    EMPTY: "Queue clear. No action required.",
    SPARSE: "Queue nearly empty. Keep the checkpoint staffed.",
    NORMAL: "Normal load. Monitor lane throughput; no action required.",
    CROWDED: "Queue building. Consider opening an additional screening lane.",
    CONGESTED: "Congestion detected. Activate overflow management and direct passengers to secondary lanes.",
    BACKLOG: "Backlog: flow below baseline with long waits. Activate the queue-management protocol.",
}

# ---------------------------------------------------------------------
#  TIME HELPERS (ISO 8601, UTC, trailing Z)
# ---------------------------------------------------------------------


def recommendation_for(state: str, config) -> str:
    """Operational recommendation text for a state (config-overridable)."""
    overrides = getattr(config, "recommendations", None) or {}
    if state in overrides:
        return str(overrides[state])
    return DEFAULT_RECOMMENDATIONS.get(state, DEFAULT_RECOMMENDATIONS[UNCONFIGURED])


def iso_from_unix(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def unix_now() -> float:
    return time.time()


def fmt_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    s = max(0, int(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------
#  GEOMETRY HELPERS (pixel space)
# ---------------------------------------------------------------------


def _in_polygon(px: float, py: float, poly) -> bool:
    if poly is None or len(poly) < 3:
        return False
    return cv2.pointPolygonTest(np.array(poly, dtype=np.float32), (float(px), float(py)), False) >= 0


def _polygon_area_px(poly) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    return float(abs(cv2.contourArea(np.array(poly, dtype=np.float32))))


def _line_side(a, b, px: float, py: float) -> float:
    """Signed cross product (b-a) x (p-a); sign selects a half-plane."""
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


def _side_int(a, b, p) -> int:
    """``_line_side`` reduced to a strict ``-1`` / ``0`` / ``+1`` sign."""
    s = _line_side(a, b, float(p[0]), float(p[1]))
    if s == 0.0:
        return 0
    return 1 if s > 0 else -1


def reference_xy(box, reference_point: str = CENTROID_REFERENCE) -> Tuple[float, float]:
    """The tracking reference point of a ``(x1, y1, x2, y2)`` bbox.

    ``bottom_center`` (the calibrated default for overhead and
    high-angle cameras: the bottom of the box is the closest estimator
    of the real footprint on the ground) or ``centroid`` (the legacy
    behaviour, retained as a configurable compatibility fallback).
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    if reference_point == BOTTOM_CENTER_REFERENCE and _MAPPING_AVAILABLE:
        return (x1 + x2) / 2.0, y2
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _segment_crosses_line(p1, p2, a, b) -> bool:
    """True if segment p1->p2 crosses the infinite line through a, b."""
    s1 = _line_side(a, b, p1[0], p1[1])
    s2 = _line_side(a, b, p2[0], p2[1])
    if s1 == 0 or s2 == 0:
        return True
    return (s1 > 0) != (s2 > 0)


# ---------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------


def _parse_rotation(raw) -> int:
    """Coerce camera.rotation to a valid 0/90/180/270 (defaults to 0)."""
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        value = -1
    if value not in VALID_ROTATIONS:
        if raw is not None:
            print(f"[WARN] camera.rotation {raw!r} invalid — using 0")
        value = 0
    return value


@dataclass
class CameraConfig:
    """Camera-relative layout + classifier thresholds."""

    width: int = 0
    height: int = 0
    display_scale: float = 1.0
    detect_mode: str = "person"

    roi: Optional[list] = None
    counting_line: Optional[list] = None
    xray_side_point: Optional[list] = None

    # ── Mapping (the single active geometry source when valid) ─────
    # Loaded from config/mappings.yaml, keyed by `camera_id`.
    # When set, `roi` / `counting_line` / `xray_side_point` above are
    # IGNORED — see `_mapping_pixel()` in QueueAnalytics.
    mapping: Optional["MappingGeometry"] = None
    # Tracking reference point. ``None`` means "not set in camera.yaml",
    # in which case the mapping's own ``reference_point`` decides
    # (``bottom_center`` for new mappings, ``centroid`` for legacy).
    reference_point: Optional[str] = None
    # Per-track/per-line crossing deduplication (frames of stability
    # required before an event can fire). Separate knob from the
    # classifier's `debounce_frames`; see DEFAULT_CROSSING_DEBOUNCE_FRAMES.
    crossing_debounce_frames: int = DEFAULT_CROSSING_DEBOUNCE_FRAMES
    # Where the active mapping came from: "mappings.yaml", "legacy"
    # (derived in-memory from camera.yaml geometry) or "none".
    mapping_source: str = "none"

    queue_capacity: int = 30
    flow_windows_minutes: list = field(default_factory=lambda: [1, 5])
    track_lost_grace_seconds: float = 1.5

    occupancy_sparse: float = 0.10
    occupancy_normal: float = 0.40
    occupancy_crowded: float = 0.60
    occupancy_congested: float = 0.80
    hysteresis: float = 0.05
    debounce_frames: int = 5
    congested_max_spacing_px: float = 60.0
    congested_require_spacing: bool = True

    backlog_min_people: int = 15
    backlog_min_occupancy: float = 0.60
    backlog_min_wait_seconds: float = 300.0
    backlog_persistence_seconds: float = 10.0
    backlog_flow_ratio_below_baseline: float = 0.5
    baseline_people_per_minute: Optional[float] = None

    homography_path: Optional[str] = None
    px_per_metre: Optional[float] = None

    source_shape_w: Optional[int] = None
    source_shape_h: Optional[int] = None
    path: Optional[Path] = None

    camera_id: str = "CAMERA 01"

    # Camera source: webcam index ("0", "1", ...), video file path,
    # image path, or RTSP/HTTP URL.  None → CLI/default behavior
    # (webcam index 0 unless --source is given).
    source: Optional[str] = None

    # Clockwise frame rotation applied at the capture edge, before
    # detection: 0, 90, 180 or 270 degrees. Changing it also rotates the
    # active calibration geometry (mapping or legacy pixels) so the same
    # physical scene region stays covered. See app/orientation.py.
    rotation: int = 0

    # Queue-movement classification: the queue is MOVING when at least
    # `movement_min_events` line crossings happen inside `movement_window_s`.
    movement_window_s: float = 60.0
    movement_min_events: int = 1

    # Rolling average of in-queue people over this window (display/charts
    # only — never feeds the classifier, so existing thresholds hold).
    occupancy_avg_window_s: float = 120.0

    # Dashboard layout knobs.
    dashboard_panel_width: int = 360
    dashboard_chart_history: int = 120

    # Per-state operational-recommendation text overrides.
    recommendations: dict = field(default_factory=dict)

    @property
    def roi_configured(self) -> bool:
        """True when the ROI polygon is usable for analytics.

        The active mapping counts, because it is the single geometry
        source; the legacy ``config.roi`` only matters when no mapping
        is loaded.
        """
        if self.mapping_active and self.mapping.roi:
            return True
        return bool(self.roi and len(self.roi) >= 3 and _polygon_area_px(self.roi) > 0)

    @property
    def line_configured(self) -> bool:
        """True when at least one counting line is usable."""
        if self.mapping_active and self.mapping.lines:
            return True
        return bool(self.counting_line and len(self.counting_line) == 2 and self.xray_side_point)

    @property
    def mapping_active(self) -> bool:
        """True when a valid mapping is loaded and drives the geometry."""
        return _MAPPING_AVAILABLE and (self.mapping is not None) and bool(self.mapping.lines)

    @property
    def geometry_configured(self) -> bool:
        """True when the ROI and at least one counting line are usable.

        Identical to the pre-mapping definition when no mapping is
        loaded, and the mapping-driven definition afterwards, so callers
        downstream of the pipeline do not have to branch on the mapping
        to know whether analytics can run.
        """
        if self.mapping_active:
            return True
        return self.roi_configured and self.line_configured

    @property
    def calibrated(self) -> bool:
        return bool(self.homography_path or self.px_per_metre)

    @property
    def baseline_available(self) -> bool:
        return self.baseline_people_per_minute is not None

    @classmethod
    def from_dict(cls, data: dict, path=None) -> "CameraConfig":
        cam = data.get("camera", {}) or {}
        cf = data.get("classifier", {}) or {}
        bl = data.get("backlog", {}) or {}
        flow = bl.get("flow", {}) or {}
        cal = data.get("calibration", {}) or {}
        src = data.get("source", {}) or {}
        mov = data.get("movement", {}) or {}
        dash = data.get("dashboard", {}) or {}
        rec = data.get("recommendations", {}) or {}
        return cls(
            width=int(cam.get("width", 0) or 0),
            height=int(cam.get("height", 0) or 0),
            display_scale=float(cam.get("display_scale", 1.0) or 1.0),
            detect_mode=str(cam.get("detect_mode", "person") or "person").lower(),
            roi=cam.get("roi"),
            counting_line=cam.get("counting_line"),
            xray_side_point=cam.get("xray_side_point"),
            queue_capacity=int(cam.get("queue_capacity", 30)),
            flow_windows_minutes=cam.get("flow_windows_minutes", [1, 5]),
            track_lost_grace_seconds=float(cam.get("track_lost_grace_seconds", 1.5)),
            occupancy_sparse=float(cf.get("occupancy_sparse", 0.10)),
            occupancy_normal=float(cf.get("occupancy_normal", 0.40)),
            occupancy_crowded=float(cf.get("occupancy_crowded", 0.60)),
            occupancy_congested=float(cf.get("occupancy_congested", 0.80)),
            hysteresis=float(cf.get("hysteresis", 0.05)),
            debounce_frames=int(cf.get("debounce_frames", 5)),
            congested_max_spacing_px=float(cf.get("congested_max_spacing_px", 60.0)),
            congested_require_spacing=bool(cf.get("congested_require_spacing", True)),
            backlog_min_people=int(bl.get("min_people", 15)),
            backlog_min_occupancy=float(bl.get("min_occupancy", 0.60)),
            backlog_min_wait_seconds=float(bl.get("min_wait_seconds", 300.0)),
            backlog_persistence_seconds=float(bl.get("persistence_seconds", 10.0)),
            backlog_flow_ratio_below_baseline=float(bl.get("flow_ratio_below_baseline", 0.5)),
            baseline_people_per_minute=flow.get("baseline_people_per_minute"),
            homography_path=cal.get("homography_path"),
            px_per_metre=cal.get("px_per_metre"),
            source_shape_w=src.get("shape_w"),
            source_shape_h=src.get("shape_h"),
            path=Path(path) if path else None,
            camera_id=str(cam.get("camera_id", "CAMERA 01") or "CAMERA 01"),
            source=str(cam["source"]) if cam.get("source") is not None else None,
            # Clockwise capture rotation; invalid values fall back to 0
            # rather than buckling the whole config load.
            rotation=_parse_rotation(cam.get("rotation", 0)),
            movement_window_s=float(mov.get("window_s", 60.0) or 60.0),
            movement_min_events=int(mov.get("min_events", 1)),
            occupancy_avg_window_s=float(dash.get("occupancy_avg_window_s", 120.0) or 120.0),
            dashboard_panel_width=int(dash.get("panel_width", 360)),
            dashboard_chart_history=int(dash.get("chart_history", 120)),
            recommendations=dict(rec),
            reference_point=str(
                data.get("mapping", {}).get("reference_point", "") or ""
            ).lower() or None,
            crossing_debounce_frames=int(
                data.get("mapping", {}).get("crossing_debounce_frames",
                                            DEFAULT_CROSSING_DEBOUNCE_FRAMES)
                or DEFAULT_CROSSING_DEBOUNCE_FRAMES
            ),
        )

    def to_dict(self) -> dict:
        return {
            "camera": {
                "width": self.width,
                "height": self.height,
                "display_scale": self.display_scale,
                "detect_mode": self.detect_mode,
                "roi": self.roi,
                "counting_line": self.counting_line,
                "xray_side_point": self.xray_side_point,
                "queue_capacity": self.queue_capacity,
                "flow_windows_minutes": self.flow_windows_minutes,
                "track_lost_grace_seconds": self.track_lost_grace_seconds,
                "camera_id": self.camera_id,
                "source": self.source,
            },
            "classifier": {
                "occupancy_sparse": self.occupancy_sparse,
                "occupancy_normal": self.occupancy_normal,
                "occupancy_crowded": self.occupancy_crowded,
                "occupancy_congested": self.occupancy_congested,
                "hysteresis": self.hysteresis,
                "debounce_frames": self.debounce_frames,
                "congested_max_spacing_px": self.congested_max_spacing_px,
                "congested_require_spacing": self.congested_require_spacing,
            },
            "backlog": {
                "min_people": self.backlog_min_people,
                "min_occupancy": self.backlog_min_occupancy,
                "min_wait_seconds": self.backlog_min_wait_seconds,
                "persistence_seconds": self.backlog_persistence_seconds,
                "flow_ratio_below_baseline": self.backlog_flow_ratio_below_baseline,
                "flow": {"baseline_people_per_minute": self.baseline_people_per_minute},
            },
            "calibration": {"homography_path": self.homography_path, "px_per_metre": self.px_per_metre},
            "source": {"shape_w": self.source_shape_w, "shape_h": self.source_shape_h},
            "movement": {"window_s": self.movement_window_s, "min_events": self.movement_min_events},
            "dashboard": {
                "occupancy_avg_window_s": self.occupancy_avg_window_s,
                "panel_width": self.dashboard_panel_width,
                "chart_history": self.dashboard_chart_history,
            },
            "recommendations": dict(self.recommendations),
        }

    def save(self, path=None):
        dest = Path(path or self.path)
        if dest is None:
            raise ValueError("no save path given")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        self.path = dest

    def validate_roi(self, shape=None) -> list:
        errors = []
        if not self.roi or len(self.roi) < 3:
            errors.append("ROI must have at least 3 polygon points")
        elif _polygon_area_px(self.roi) <= 0:
            errors.append("ROI polygon is degenerate (zero area)")
        if shape and self.roi:
            w, h = shape[1], shape[0]
            for x, y in self.roi:
                if not (0 <= x <= w and 0 <= y <= h):
                    errors.append(f"ROI point ({x}, {y}) is outside the frame")
        return errors

    def validate_line(self, shape=None) -> list:
        errors = []
        if not self.counting_line or len(self.counting_line) != 2:
            errors.append("counting_line must have exactly 2 endpoints")
        if not self.xray_side_point:
            errors.append("xray_side_point is required")
        if shape:
            w, h = shape[1], shape[0]
            for name, pts in (
                ("counting_line", self.counting_line or []),
                ("xray_side_point", [self.xray_side_point] if self.xray_side_point else []),
            ):
                for x, y in pts:
                    if not (0 <= x <= w and 0 <= y <= h):
                        errors.append(f"{name} point ({x}, {y}) is outside the frame")
        return errors


def load_camera_config(path) -> CameraConfig:
    """Load a camera config; returns defaults if the file is missing.

    Mapping resolution order (the mapping is the single active geometry
    source when one is valid):

      1. ``config/mappings.yaml`` entry for ``camera_id``  → mapping
      2. legacy ``camera.yaml`` ROI + counting_line + xray_side_point
         → derived mapping (in-memory only, never persisted back)

    If the loaded mapping fails validation, the pipeline falls back to
    legacy geometry and records why, so a broken mapping can never
    silently blank the analytics.
    """
    path = Path(path)
    if not path.exists():
        return CameraConfig(path=path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    config = CameraConfig.from_dict(data, path=path)
    return _resolve_camera_mapping(config, mapping_path=None)


def _resolve_camera_mapping(config: "CameraConfig", mapping_path=None) -> "CameraConfig":
    """Attach a validated mapping to ``config``.

    Loads it from ``mappings.yaml`` when present, otherwise derives one
    from the legacy ``counting_line`` + ``xray_side_point`` geometry.
    Invalid mappings are dropped (with a log line) so that the analytics
    engine keeps running the previous geometry instead of blanking.
    """
    if not _MAPPING_AVAILABLE:
        return config
    if mapping_path is None:
        try:
            from mapping import default_mappings_path
            mapping_path = default_mappings_path(config.path)
        except Exception:
            mapping_path = None
    try:
        from mapping import load_mapping, from_legacy_camera_config
    except ImportError:
        return config

    camera_id = config.camera_id or "CAMERA 01"
    shape = None
    if config.source_shape_w and config.source_shape_h:
        shape = (int(config.source_shape_w), int(config.source_shape_h))

    mapping = None
    source = "legacy camera.yaml geometry"
    if mapping_path is not None:
        try:
            mapping = load_mapping(Path(mapping_path), camera_id=camera_id)
        except MappingError as exc:
            print(f"[WARN] mappings.yaml unusable, ignoring: {exc}")
            mapping = None
        if mapping is not None:
            source = f"{Path(mapping_path).name}"

    if mapping is None and config.roi_configured and config.line_configured and shape:
        # Legacy fallback: derive an in-memory mapping, never persisted.
        mapping = from_legacy_camera_config(
            config.roi,
            config.counting_line,
            config.xray_side_point,
            shape=shape,
            camera_id=camera_id,
            reference_point=CENTROID_REFERENCE,
        )

    if mapping is None:
        config.mapping_source = "none"
        config.mapping = None
        return config

    errors = validate_mapping_geometry(mapping, shape=mapping.source_shape)
    if errors:
        print(
            f"[WARN] mapping {camera_id!r} from {source} is invalid and inactive: "
            + "; ".join(errors)
        )
        config.mapping = None
        config.mapping_source = "none"
        return config

    if config.reference_point:
        # camera.yaml explicitly sets it (compatibility switch); the
        # mapping's own value only applies when camera.yaml does not.
        mapping.reference_point = config.reference_point
    config.mapping = mapping
    config.mapping_source = "legacy" if source.startswith("legacy") else "mappings.yaml"
    print(
        f"[INFO] mapping {camera_id!r} active from {source} — "
        f"roi={len(mapping.roi)}pt, lines={sorted(mapping.lines)}, "
        f"ref={mapping.reference_point}"
    )
    return config


# ---------------------------------------------------------------------
#  EVENT LOG
# ---------------------------------------------------------------------


class EventLog:
    """In-memory event buffer with optional JSONL persistence."""

    def __init__(self, record_path=None):
        self.record_path = Path(record_path) if record_path else None
        self.events = []
        self._fh = None
        if self.record_path:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.record_path, "w")

    def add(self, event: dict):
        self.events.append(event)
        if self._fh:
            self._fh.write(json.dumps(event, default=str) + "\n")
            self._fh.flush()

    def extend(self, events):
        for e in events:
            self.add(e)


# ---------------------------------------------------------------------
#  PER-PERSON STATE
# ---------------------------------------------------------------------


@dataclass
class _TrackLineState:
    """Per-track, per-line crossing state.

    ``side`` is the last committed half-plane side of the track relative
    to the line (``+1`` / ``-1`` for the two sides, ``0`` until the first
    observation). ``pending_side`` and ``pending_frames`` accumulate a
    candidate crossing until the track has been on the new side for
    ``crossing_debounce_frames`` consecutive frames — that is what makes
    jitter next to a line produce a single event instead of a burst.
    """

    side: int = 0
    pending_side: int = 0
    pending_frames: int = 0
    last_event_unix: Optional[float] = None
    last_direction: Optional[str] = None
    reverse_events: int = 0

    def reset(self) -> None:
        self.side = 0
        self.pending_side = 0
        self.pending_frames = 0
        self.last_event_unix = None
        self.last_direction = None
        self.reverse_events = 0


@dataclass
class PersonState:
    track_id: int
    first_seen_unix: float
    first_seen_iso: str
    last_seen_unix: float = 0.0
    entry_unix: Optional[float] = None
    entry_iso: Optional[str] = None
    exit_unix: Optional[float] = None
    exit_iso: Optional[str] = None
    total_waiting_s: Optional[float] = None
    prev_centroid: Optional[tuple] = None
    centroid: Optional[tuple] = None
    inside_roi: bool = False
    crossed_line: bool = False
    conf: float = 0.0
    # Per-line crossing state, keyed by line id ("in" / "out" / "legacy").
    line_states: Dict[str, _TrackLineState] = field(default_factory=dict)
    # The reference point actually used for the last crossing/ROI test —
    # kept alongside ``centroid`` so geometry regressions are visible.
    last_ref_point: Tuple[float, float] = (0.0, 0.0)


# ---------------------------------------------------------------------
#  RULE-BASED CLASSIFIER (transparent, no ML)
# ---------------------------------------------------------------------


class RuleClassifier:
    """Multi-variable congestion classifier with hysteresis + debounce.

    Every classification returns the exact reasons that produced the
    state, so behaviour can be inspected and calibrated.
    """

    def __init__(self, config: CameraConfig):
        self.cfg = config
        self._candidate = None
        self._count = 0
        self._committed = UNCONFIGURED
        self._committed_reasons: list = []
        self._backlog_since = None

    def reset(self):
        self._candidate = None
        self._count = 0
        self._committed = UNCONFIGURED
        self._committed_reasons = []
        self._backlog_since = None

    def classify(self, occ_ratio, spacing_px, count, mean_wait_s, flow_current, now_unix):
        """Return (committed_state, reasons, candidate_state, backlog_info)."""
        backlog_info = self._backlog_status(occ_ratio, count, mean_wait_s, flow_current, now_unix)
        if backlog_info.get("fire"):
            candidate, reasons = BACKLOG, backlog_info["reasons"]
        else:
            candidate, reasons = self._ladder(occ_ratio, spacing_px)

        if candidate == self._candidate:
            self._count += 1
        else:
            self._candidate = candidate
            self._count = 1

        if self._count >= self.cfg.debounce_frames:
            reasons = list(reasons)
            if "condition_sustained" not in reasons:
                reasons.append("condition_sustained")
            self._committed = candidate
            self._committed_reasons = reasons
        return self._committed, list(self._committed_reasons), candidate, backlog_info

    def _backlog_status(self, occ_ratio, count, mean_wait_s, flow_current, now_unix):
        if not self.cfg.baseline_available:
            return {
                "check": "UNAVAILABLE",
                "fire": False,
                "reasons": [],
                "sustained_s": 0.0,
                "message": BACKLOG_UNAVAILABLE_MSG,
            }
        reasons = []
        if count >= self.cfg.backlog_min_people:
            reasons.append("high_queue_count")
        if occ_ratio >= self.cfg.backlog_min_occupancy:
            reasons.append("high_occupancy")
        if flow_current is not None and flow_current < (
            self.cfg.baseline_people_per_minute * self.cfg.backlog_flow_ratio_below_baseline
        ):
            reasons.append("flow_significantly_below_baseline")
        if mean_wait_s is not None and mean_wait_s >= self.cfg.backlog_min_wait_seconds:
            reasons.append("long_waiting_time")
        required = {"high_queue_count", "high_occupancy", "flow_significantly_below_baseline", "long_waiting_time"}
        if required.issubset(set(reasons)):
            if self._backlog_since is None:
                self._backlog_since = now_unix
            sustained = now_unix - self._backlog_since
            if sustained >= self.cfg.backlog_persistence_seconds:
                reasons.append("condition_sustained")
                return {"check": "AVAILABLE", "fire": True, "reasons": reasons, "sustained_s": sustained, "message": ""}
        else:
            self._backlog_since = None
        return {
            "check": "AVAILABLE",
            "fire": False,
            "reasons": reasons,
            "sustained_s": (now_unix - self._backlog_since) if self._backlog_since else 0.0,
            "message": "",
        }

    def _ladder(self, occ_ratio, spacing_px):
        th = self.cfg
        downgrading = self._committed in (SPARSE, NORMAL, CROWDED, CONGESTED, BACKLOG)
        if downgrading:
            c = th.occupancy_congested * (1 - th.hysteresis)
            r = th.occupancy_crowded * (1 - th.hysteresis)
            n = th.occupancy_normal * (1 - th.hysteresis)
            s = th.occupancy_sparse * (1 - th.hysteresis)
        else:
            c, r, n, s = th.occupancy_congested, th.occupancy_crowded, th.occupancy_normal, th.occupancy_sparse
        if occ_ratio >= c:
            tier = CONGESTED
        elif occ_ratio >= r:
            tier = CROWDED
        elif occ_ratio >= n:
            tier = NORMAL
        elif occ_ratio >= s:
            tier = SPARSE
        else:
            tier = EMPTY

        if tier == CONGESTED:
            if th.congested_require_spacing and spacing_px is not None and spacing_px > th.congested_max_spacing_px:
                return CROWDED, ["queue_occupancy_above_congested_threshold", "spacing_above_congested_threshold"]
            reasons = ["queue_occupancy_above_congested_threshold"]
            if th.congested_require_spacing and spacing_px is not None:
                reasons.append("spacing_below_threshold")
            return CONGESTED, reasons
        if tier == CROWDED:
            return CROWDED, ["queue_occupancy_above_crowded_threshold"]
        if tier == NORMAL:
            return NORMAL, ["queue_occupancy_above_normal_threshold"]
        if tier == SPARSE:
            return SPARSE, ["queue_occupancy_above_sparse_threshold"]
        return EMPTY, ["queue_occupancy_below_sparse_threshold"]


# ---------------------------------------------------------------------
#  CONGESTION STOPWATCH
# ---------------------------------------------------------------------


class CongestionStopwatch:
    """Tracks real congestion start/end timestamps (backend-side)."""

    def __init__(self):
        self.active = False
        self.start_unix = None
        self.start_iso = None
        self.duration_s = 0.0

    def update(self, state, now_unix):
        events = []
        severe = state in SEVERE_STATES
        if severe and not self.active:
            self.active = True
            self.start_unix = now_unix
            self.start_iso = iso_from_unix(now_unix)
            self.duration_s = 0.0
            events.append({
                "event_type": "congestion_start",
                "event_time": self.start_iso,
                "state": state,
                "label": "Congestion detected",
            })
        elif severe and self.active:
            self.duration_s = now_unix - self.start_unix
        elif not severe and self.active:
            self.duration_s = now_unix - self.start_unix
            events.append({
                "event_type": "congestion_end",
                "event_time": iso_from_unix(now_unix),
                "started_at": self.start_iso,
                "duration_s": round(self.duration_s, 3),
                "ended_state": state,
                "label": f"Congestion cleared after {fmt_duration(self.duration_s)}",
            })
            self.active = False
            self.start_unix = None
            self.start_iso = None
        return events

    def to_dict(self):
        return {
            "active": self.active,
            "duration_s": round(self.duration_s, 3),
            "started_at": self.start_iso if self.active else None,
            "ended_at": None,
        }



@dataclass
class _LostTrackRecord:
    """Track information cached briefly when a track is lost inside the queue ROI."""
    track_id: int
    lost_unix: float
    entry_unix: Optional[float]
    entry_iso: Optional[str]
    first_seen_unix: float
    first_seen_iso: str
    last_ref_point: Tuple[float, float]
    conf: float


class ServiceTimeTracker:
    """Tracks inter-departure service time (Delta t) at the X-ray screening exit.

    Applies Little's Law and queue service rate smoothing:
    - Delta t = t_exit - t_prev_exit represents actual service duration per passenger.
    - An Exponential Moving Average (EMA) filters noise while adapting to real flow.
    - Estimated Wait Time: W_q = L_q * S_eff
    - Estimated Queue Clearance Time: T_clear = L_q * S_eff
    """

    def __init__(self, baseline_service_s: float = 20.0, alpha: float = 0.25):
        self.baseline_service_s = float(baseline_service_s)
        self.alpha = float(alpha)
        self.smoothed_service_s = float(baseline_service_s)
        self.last_exit_unix: Optional[float] = None
        self.exit_history: deque = deque(maxlen=20)
        self.total_completed = 0

    def record_exit(self, now_unix: float) -> float:
        """Record an exit event and update smoothed service time."""
        self.total_completed += 1
        delta = None
        if self.last_exit_unix is not None:
            raw_delta = max(2.0, min(now_unix - self.last_exit_unix, 180.0))
            delta = raw_delta
            self.smoothed_service_s = (self.alpha * delta) + ((1.0 - self.alpha) * self.smoothed_service_s)
            self.exit_history.append((now_unix, delta))
        self.last_exit_unix = now_unix
        return delta or self.smoothed_service_s

    def effective_service_time_s(self, now_unix: float, queue_count: int) -> float:
        """Effective service time with graceful decay if passenger is taking longer."""
        base = self.smoothed_service_s
        if queue_count > 0 and self.last_exit_unix is not None:
            elapsed = now_unix - self.last_exit_unix
            if elapsed > base:
                base = min(base * 0.7 + elapsed * 0.3, 120.0)
        return max(3.0, round(base, 1))

    def estimate_wait_seconds(self, queue_count: int, now_unix: float) -> float:
        """Estimated wait time for the next joining passenger: W_q = L_q * S_eff."""
        if queue_count <= 0:
            return 0.0
        eff = self.effective_service_time_s(now_unix, queue_count)
        return round(queue_count * eff, 1)

    def estimate_clearance_seconds(self, queue_count: int, now_unix: float) -> float:
        """Estimated time until the current queue is fully cleared."""
        return self.estimate_wait_seconds(queue_count, now_unix)

    def pacing_status(self, queue_count: int, now_unix: float) -> str:
        """Pacing index: OPTIMAL | STEADY | SLOW_INSPECTION | STAGNANT | IDLE."""
        if queue_count <= 0:
            return "IDLE"
        if self.last_exit_unix is not None and (now_unix - self.last_exit_unix) > 90.0:
            return "STAGNANT"
        eff = self.effective_service_time_s(now_unix, queue_count)
        if eff <= 15.0:
            return "OPTIMAL"
        if eff > 35.0:
            return "SLOW_INSPECTION"
        return "STEADY"

    def smoothed_rate_per_min(self, now_unix: float, queue_count: int) -> float:
        """Throughput / clearance rate in passengers per minute."""
        if queue_count <= 0 and (self.last_exit_unix is None or now_unix - self.last_exit_unix > 120.0):
            return 0.0
        eff = self.effective_service_time_s(now_unix, queue_count)
        return round(60.0 / max(eff, 1.0), 1)

    def smoothed_rate_per_hour(self, now_unix: float, queue_count: int) -> int:
        """Throughput in passengers per hour."""
        rate_min = self.smoothed_rate_per_min(now_unix, queue_count)
        return int(round(rate_min * 60.0))


# ---------------------------------------------------------------------
#  ANALYTICS ENGINE
# ---------------------------------------------------------------------


class QueueAnalytics:
    """Per-frame camera queue analytics.

    Consumes ``tracks`` (x1, y1, x2, y2, track_id) and produces a
    JSON-serialisable snapshot plus an event stream. All measurements are
    pixel-relative unless the camera is calibrated.
    """

    def __init__(self, config: CameraConfig, record_path=None):
        self.config = config
        self.persons: dict = {}
        self.exit_timestamps: deque = deque()
        self.entry_timestamps: deque = deque()
        self.total_entries = 0
        self.total_exits = 0
        self.classifier = RuleClassifier(config)
        self.stopwatch = CongestionStopwatch()
        self.event_log = EventLog(record_path)
        self.frame_count = 0
        self._prev_state = UNCONFIGURED
        self._state_start_unix: Optional[float] = None
        self.occupancy_series: deque = deque()
        # Airport X-Ray Service Rate & Little's Law Tracker
        self.service_tracker = ServiceTimeTracker(
            baseline_service_s=(
                60.0 / self.config.baseline_people_per_minute
                if getattr(self.config, "baseline_people_per_minute", None)
                else 20.0
            )
        )
        # Brief spatial memory for tracks temporarily lost in ROI (occlusion recovery)
        self._lost_roi_tracks: Dict[int, _LostTrackRecord] = {}
        # Resolved per-frame geometry (ROI in native pixels, counting
        # lines in native pixels, tracking reference point). Recomputed
        # on the first track of the frame, cached for the rest of it, so
        # ROI membership and crossings always use the same coordinates.
        self._geom_cache_shape: Optional[Tuple[int, int]] = None
        self._geom_cache: Optional[dict] = None

    # ── resolved geometry (single active source of truth) ─────

    def _geometry_for_shape(self, shape: Tuple[int, int]) -> dict:
        """Resolve the ROI polygon and counting lines for one frame.

        Returns ``{"roi": [(x, y) ...], "lines": {id: (a, b, direction)},
        "ref": "bottom_center"|"centroid", "configured": bool}``.

        A valid mapping is the single active geometry source: when one
        is loaded, the legacy ``config.roi`` / ``config.counting_line`` /
        ``config.xray_side_point`` are ignored entirely. When no mapping
        is loaded the legacy geometry is used with the legacy semantics
        (crossing towards ``xray_side_point`` = EXIT, away = ENTER) and
        the centroid reference point, exactly as before this refactor.

        Results are cached per-frame SHAPE, so repeated calls inside the
        same frame (one per track) never re-run the conversion.
        """
        w, h = shape
        if w <= 0 or h <= 0:
            return {"roi": (), "lines": {},
                    "ref": self.config.reference_point or CENTROID_REFERENCE,
                    "configured": False}
        if self._geom_cache_shape == shape and self._geom_cache is not None:
            return self._geom_cache

        cfg = self.config
        if cfg.mapping_active and MappingGeometry is not None:
            try:
                pix = mapping_to_pixel(cfg.mapping, (w, h))
                lines = {}
                if pix.in_line is not None:
                    lines["in"] = (pix.in_line[0], pix.in_line[1], pix.in_direction or "IN")
                if pix.out_line is not None:
                    lines["out"] = (pix.out_line[0], pix.out_line[1], pix.out_direction or "OUT")
                resolved = {
                    "roi": tuple(pix.roi),
                    "lines": lines,
                    "ref": pix.reference_point,
                    "configured": bool(pix.roi) and bool(lines),
                }
                self._geom_cache_shape = (w, h)
                self._geom_cache = resolved
                return resolved
            except MappingError:
                pass  # fall through to the legacy geometry below

        # ── legacy path, semantics unchanged ─────────────────
        roi = tuple((float(x), float(y)) for x, y in (cfg.roi or ()))
        lines = {}
        if cfg.counting_line and len(cfg.counting_line) == 2 and cfg.xray_side_point:
            a = (float(cfg.counting_line[0][0]), float(cfg.counting_line[0][1]))
            b = (float(cfg.counting_line[1][0]), float(cfg.counting_line[1][1]))
            side = (float(cfg.xray_side_point[0]), float(cfg.xray_side_point[1]))
            # Legacy semantics: crossing *towards* `xray_side_point` is an
            # EXIT, away from it is an ENTER. `out` therefore must carry
            # its positive half-plane on the X-ray side. Which ordering
            # that is depends on the geometry (horizontal or vertical
            # line), so derive it from the sign of `_line_side` at the
            # X-ray point instead of assuming an orientation:
            if _line_side(a, b, side[0], side[1]) >= 0:
                lines["out"] = (a, b, "OUT")
                lines["in"] = (b, a, "IN")
            else:
                lines["out"] = (b, a, "OUT")
                lines["in"] = (a, b, "IN")
        resolved = {
            "roi": roi,
            "lines": lines,
            "ref": CENTROID_REFERENCE,
            "configured": (len(roi) >= 3) and bool(lines),
        }
        self._geom_cache_shape = (w, h)
        self._geom_cache = resolved
        return resolved

    # ── per-frame entry point ────────────────────────────────

    def update(self, tracks, shape, now_unix=None, track_confs=None) -> dict:
        now_unix = now_unix if now_unix is not None else unix_now()
        now_iso = iso_from_unix(now_unix)
        self.frame_count += 1
        h, w = shape
        self._geom_cache_shape = None  # invalidate per-frame geometry cache
        self._geom_cache = None
        self._geom_cache_shape = (int(w), int(h))
        events = []
        track_confs = track_confs or {}

        live_ids = {int(t[4]) for t in tracks}
        events += self._prune_lost(live_ids, now_unix)

        centroids = {}
        for t in tracks:
            tid = int(t[4])
            x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            centroids[tid] = (cx, cy)

            geom = self._geometry_for_shape((w, h))
            ref_point = reference_xy((x1, y1, x2, y2), geom["ref"])

            ps = self.persons.get(tid)
            if ps is None:
                # Check spatial continuity with lost ROI tracks (occlusion recovery)
                inside_candidate = _in_polygon(ref_point[0], ref_point[1], geom["roi"]) if geom["roi"] else False
                if inside_candidate and self._lost_roi_tracks:
                    best_tid = None
                    min_dist = float("inf")
                    for l_tid, l_rec in list(self._lost_roi_tracks.items()):
                        if now_unix - l_rec.lost_unix <= (self.config.track_lost_grace_seconds * 3.0):
                            d = math.hypot(ref_point[0] - l_rec.last_ref_point[0], ref_point[1] - l_rec.last_ref_point[1])
                            if d < min_dist and d <= 90.0:
                                min_dist = d
                                best_tid = l_tid
                    if best_tid is not None:
                        l_rec = self._lost_roi_tracks.pop(best_tid)
                        ps = PersonState(
                            track_id=tid,
                            first_seen_unix=l_rec.first_seen_unix,
                            first_seen_iso=l_rec.first_seen_iso,
                            entry_unix=l_rec.entry_unix,
                            entry_iso=l_rec.entry_iso,
                            inside_roi=True,
                        )
                        self.persons[tid] = ps

                if ps is None:
                    ps = PersonState(track_id=tid, first_seen_unix=now_unix, first_seen_iso=now_iso)
                    self.persons[tid] = ps
            ps.last_seen_unix = now_unix
            ps.conf = track_confs.get(tid, ps.conf)
            ps.centroid = (cx, cy)
            ps.last_ref_point = ref_point

            self._handle_crossing(ps, ref_point, now_unix, events, geom)

            inside = _in_polygon(ref_point[0], ref_point[1], geom["roi"]) if geom["roi"] else False
            ps.inside_roi = inside
            if inside and ps.entry_unix is None:
                ps.entry_unix = now_unix
                ps.entry_iso = now_iso
                events.append({
                    "event_type": "track_enter_roi",
                    "track_id": tid,
                    "event_time": now_iso,
                    "confidence": ps.conf,
                    "label": f"Joined queue (person #{tid})",
                })
            ps.prev_centroid = ref_point

        # ── metrics ──────────────────────────────────────────
        queue_members = [ps for ps in self.persons.values() if ps.inside_roi and ps.exit_unix is None]
        occupancy = len(queue_members)
        occ_ratio = occupancy / self.config.queue_capacity if self.config.queue_capacity > 0 else 0.0
        occ_pct = round(occ_ratio * 100.0, 1)

        waits = [now_unix - ps.entry_unix for ps in queue_members if ps.entry_unix is not None]
        avg_wait = float(np.mean(waits)) if waits else 0.0
        longest_wait = float(np.max(waits)) if waits else 0.0

        pts = [ps.centroid for ps in queue_members if ps.centroid is not None]
        spacing = self._pairwise_spacing(pts)

        flow = self._flow(now_unix)

        area_px = _polygon_area_px(self.config.roi)
        raw_density = occupancy / area_px if area_px > 0 else 0.0

        # ── classify ─────────────────────────────────────────
        if self.config.roi_configured:
            state, reasons, candidate, backlog_info = self.classifier.classify(
                occ_ratio,
                spacing.get("average_px") if spacing else None,
                occupancy,
                avg_wait,
                flow.get("current_per_min"),
                now_unix,
            )
        else:
            state, reasons, candidate = UNCONFIGURED, ["roi_not_configured"], UNCONFIGURED
            backlog_info = self.classifier._backlog_status(occ_ratio, occupancy, avg_wait, flow.get("current_per_min"), now_unix)

        # ── state-change event + duration (persistent, hysteresis-driven) ──
        state_duration_s = 0.0
        if self._state_start_unix is None:
            self._prev_state = state
            self._state_start_unix = now_unix
        elif state != self._prev_state:
            events.append({
                "event_type": "state_change",
                "event_time": now_iso,
                "from_state": self._prev_state,
                "to_state": state,
                "label": f"Status {self._prev_state} → {state}",
            })
            self._prev_state = state
            self._state_start_unix = now_unix
        state_duration_s = now_unix - (self._state_start_unix or now_unix)

        # ── rolling average of in-queue people (charts only) ──
        self.occupancy_series.append((now_unix, float(occupancy)))
        while self.occupancy_series and self.occupancy_series[0][0] < now_unix - self.config.occupancy_avg_window_s:
            self.occupancy_series.popleft()
        occ_avg_people = float(np.mean([v for _, v in self.occupancy_series])) if self.occupancy_series else float(occupancy)
        occ_avg_pct = round(occ_avg_people / self.config.queue_capacity * 100.0, 1) if self.config.queue_capacity > 0 else 0.0

        events += self.stopwatch.update(state, now_unix)
        self.event_log.extend(events)

        pred_wait = self.service_tracker.estimate_wait_seconds(occupancy, now_unix)
        clearance_s = self.service_tracker.estimate_clearance_seconds(occupancy, now_unix)
        pacing = self.service_tracker.pacing_status(occupancy, now_unix)
        eff_service_s = self.service_tracker.effective_service_time_s(now_unix, occupancy)

        return {
            "timestamp": now_iso,
            "frame_count": self.frame_count,
            "state": state,
            "reasons": reasons,
            "candidate_state": candidate,
            "backlog": backlog_info,
            "config_status": {
                "roi_configured": self.config.roi_configured,
                "line_configured": self.config.line_configured,
                "classifier_enabled": self.config.roi_configured,
                "calibrated": self.config.calibrated,
                "mapping_active": bool(self.config.mapping_active),
                "reference_point": self._geometry_for_shape((w, h)).get("ref"),
            },
            "people_detected": len(tracks),
            "people_in_queue": occupancy,
            "queue_occupancy_percent": occ_pct,
            "occupancy_avg": {
                "window_s": self.config.occupancy_avg_window_s,
                "people": round(occ_avg_people, 2),
                "percent": occ_avg_pct,
            },
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "movement": self._movement(occupancy, now_unix),
            "state_duration_s": round(state_duration_s, 3),
            "flow": flow,
            "waiting": {
                "average_s": round(avg_wait, 3),
                "longest_s": round(longest_wait, 3),
                "predicted_s": round(pred_wait, 1),
                "clearance_s": round(clearance_s, 1),
                "active": occupancy,
            },
            "pacing": pacing,
            "service_time_s": eff_service_s,
            "spacing": spacing,
            "raw_density": {
                "people_per_px2": raw_density,
                "calibrated": False,
                "label": PIXEL_DENSITY_LABEL,
            },
            "congestion": self.stopwatch.to_dict(),
            "recommendation": recommendation_for(state, self.config),
            "events_since_last": events,
        }

    # ── helpers ─────────────────────────────────────────────

    def _prune_lost(self, live_ids, now_unix) -> list:
        events = []
        for tid in list(self.persons.keys()):
            if tid in live_ids:
                continue
            ps = self.persons[tid]
            if now_unix - ps.last_seen_unix > self.config.track_lost_grace_seconds:
                if ps.entry_unix is not None and ps.exit_unix is None:
                    events.append({
                        "event_type": "track_lost",
                        "track_id": tid,
                        "event_time": iso_from_unix(now_unix),
                        "waiting_s": round(now_unix - ps.entry_unix, 3),
                        "inside_roi_at_loss": ps.inside_roi,
                        "label": f"Tracking lost (person #{tid})",
                    })
                    if ps.inside_roi:
                        self._lost_roi_tracks[tid] = _LostTrackRecord(
                            track_id=tid,
                            lost_unix=now_unix,
                            entry_unix=ps.entry_unix,
                            entry_iso=ps.entry_iso,
                            first_seen_unix=ps.first_seen_unix,
                            first_seen_iso=ps.first_seen_iso,
                            last_ref_point=ps.last_ref_point,
                            conf=ps.conf,
                        )
                del self.persons[tid]
        # Clean stale records from _lost_roi_tracks
        max_lost_age = max(5.0, self.config.track_lost_grace_seconds * 3.0)
        for old_tid in list(self._lost_roi_tracks.keys()):
            if now_unix - self._lost_roi_tracks[old_tid].lost_unix > max_lost_age:
                del self._lost_roi_tracks[old_tid]
        return events

    def _handle_crossing(self, ps, ref_point, now_unix, events, geom=None):
        """Directional per-line crossing screening for one track.

        For each configured counting line the track keeps a committed
        half-plane ``side``. A candidate crossing is a side change; it
        only becomes an event once the track has stayed on the new side
        for ``crossing_debounce_frames`` consecutive frames, which is
        what suppresses jitter right next to the line and prevents a
        track from generating duplicates. Reverse crossings (back out of
        the side just committed) are always reported explicitly and never
        increment the directional counters.

        OUT events finalise waiting time and the exit timestamp; IN
        events only register entry. Occupancy remains a live count of
        ROI members and never derives from these counters.
        """
        if geom is None:
            geom = self._geometry_for_shape(self._last_shape())
        if not geom or not geom.get("lines"):
            return
        lines = geom["lines"]
        if ps.prev_centroid is None:
            # First observation of the track: establish the side, no event.
            for ident, (a, b, _direction) in lines.items():
                side = _side_int(a, b, ref_point)
                st = ps.line_states.setdefault(ident, _TrackLineState())
                if st.side == 0:
                    st.side = side
                    st.pending_side = side
                    st.pending_frames = 0
            return

        p1, p2 = ps.prev_centroid, ref_point
        debounce = max(0, int(self.config.crossing_debounce_frames))
        any_event = False

        for ident, (a, b, direction) in lines.items():
            st = ps.line_states.setdefault(ident, _TrackLineState())
            raw_side = _side_int(a, b, p2)
            prev_raw = _side_int(a, b, p1)

            # Initialise the committed side on the first usable frame.
            if st.side == 0:
                st.side = raw_side or prev_raw
                st.pending_side = st.side
                st.pending_frames = 0
                continue

            if raw_side == 0:
                # Exactly on the line: never commits a side, never fires.
                st.pending_side = st.side
                st.pending_frames = 0
                continue

            if raw_side == st.side:
                # Still on the committed side: the pending candidate, if
                # any, is dead (it never survived the debounce window).
                st.pending_side = st.side
                st.pending_frames = 0
                continue

            # Candidate crossing: the track is on the far side.
            if raw_side != st.pending_side:
                st.pending_side = raw_side
                st.pending_frames = 1
            else:
                st.pending_frames += 1
            if st.pending_frames < debounce:
                continue  # hold for `debounce` consecutive frames first

            # Committed crossing.
            st.side = raw_side
            st.pending_side = raw_side
            st.pending_frames = 0
            st.last_event_unix = now_unix
            st.last_direction = direction
            event = self._emit_line_event(ps, ident, direction, raw_side, now_unix)
            if event is not None:
                events.append(event)
                any_event = True

        if any_event:
            ps.crossed_line = True

    def _emit_line_event(self, ps, ident, direction, raw_side, now_unix) -> Optional[dict]:
        """Produce the wire event for a committed line crossing.

        A crossing is *forward* when the track moves onto the side the
        line's direction arrow points into and *reversed* when it moves
        back out of it. Only OUT forward crossings finalise waiting time.

        Wire contract: ``direction`` keeps the historical ``ENTER`` /
        ``EXIT`` values (``app/snapshot.py`` and the HUD colour coding
        depend on them); ``line_direction`` carries the mapping schema's
        raw ``IN`` / ``OUT`` value.
        """
        st = ps.line_states[ident]
        forward = raw_side > 0
        event_type = "line_crossing" if forward else "line_crossing_reversed"
        if not forward:
            st.reverse_events += 1
        wire_direction = "EXIT" if direction == "OUT" else "ENTER"

        waiting_s = None
        label = None
        if forward:
            if wire_direction == "EXIT":
                ps.exit_unix = now_unix
                ps.exit_iso = iso_from_unix(now_unix)
                self.total_exits += 1
                self.exit_timestamps.append(now_unix)
                self.service_tracker.record_exit(now_unix)
                label = f"Cleared screening (person #{ps.track_id})"
            else:
                ps.entry_unix = now_unix
                ps.entry_iso = iso_from_unix(now_unix)
                self.total_entries += 1
                self.entry_timestamps.append(now_unix)
                label = f"Entered queue (person #{ps.track_id})"
        # Waiting time is only meaningful once the track has an entry.
        if ps.entry_unix is not None:
            waiting_s = round(now_unix - ps.entry_unix, 3)
        if forward and wire_direction == "EXIT":
            ps.total_waiting_s = waiting_s

        return {
            "event_type": event_type,
            "line_id": ident,
            "direction": wire_direction,
            "line_direction": direction,
            "reversed": not forward,
            "track_id": ps.track_id,
            "event_time": iso_from_unix(now_unix),
            "waiting_s": waiting_s,
            "confidence": ps.conf,
            "label": label or (
                # Operator language: a person moving back across a line.
                # The engineering wording stays on the technical field.
                "Passenger moved back across the "
                + ("entry" if wire_direction == "ENTER" else "exit")
                + " line"
            ),
        }

    def _last_shape(self) -> Tuple[int, int]:
        """Frame shape cached by the most recent ``update()`` call."""
        return self._geom_cache_shape or (0, 0)

    def reset_mapping(self) -> None:
        """Drop cached geometry and per-track line state (config reload).

        Called when a new mapping is applied at runtime so no stale
        ROI/line cache or per-track side memory survives the swap.
        """
        self._geom_cache_shape = None
        self._geom_cache = None
        for ps in self.persons.values():
            ps.line_states.clear()

    def _flow(self, now_unix) -> dict:
        windows = self.config.flow_windows_minutes or [1, 5]
        cutoff = now_unix - max(windows) * 60
        while self.exit_timestamps and self.exit_timestamps[0] < cutoff:
            self.exit_timestamps.popleft()
        while self.entry_timestamps and self.entry_timestamps[0] < cutoff:
            self.entry_timestamps.popleft()
        out = {"current_per_min": 0, "entry_current_per_min": 0}
        out["current_per_min"] = sum(1 for ts in self.exit_timestamps if ts >= now_unix - 60)
        out["entry_current_per_min"] = sum(1 for ts in self.entry_timestamps if ts >= now_unix - 60)
        for m in windows:
            if m == 1:
                continue
            out[f"avg_{m}min_per_min"] = round(sum(1 for ts in self.exit_timestamps if ts >= now_unix - m * 60) / m, 3)
            out[f"entry_avg_{m}min_per_min"] = round(sum(1 for ts in self.entry_timestamps if ts >= now_unix - m * 60) / m, 3)
        queue_count = sum(1 for ps in self.persons.values() if ps.inside_roi and ps.exit_unix is None)
        out["smoothed_rate_per_min"] = self.service_tracker.smoothed_rate_per_min(now_unix, queue_count)
        out["smoothed_rate_per_hour"] = self.service_tracker.smoothed_rate_per_hour(now_unix, queue_count)
        return out

    def _movement(self, occupancy: int, now_unix) -> str:
        """MOVING / STAGNANT / IDLE from recent line crossings."""
        cutoff = now_unix - self.config.movement_window_s
        events = sum(
            1 for ts in self.entry_timestamps if ts >= cutoff
        ) + sum(
            1 for ts in self.exit_timestamps if ts >= cutoff
        )
        if occupancy <= 0:
            return "IDLE"
        if events >= self.config.movement_min_events:
            return "MOVING"
        return "STAGNANT"

    @staticmethod
    def _pairwise_spacing(pts):
        if len(pts) < 2:
            return {"average_px": None, "minimum_px": None, "median_px": None, "pairs": 0}
        arr = np.array(pts, dtype=float)
        d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
        vals = d[np.triu_indices(len(arr), k=1)]
        return {
            "average_px": round(float(vals.mean()), 3),
            "minimum_px": round(float(vals.min()), 3),
            "median_px": round(float(np.median(vals)), 3),
            "pairs": int(len(vals)),
        }


# ---------------------------------------------------------------------
#  OVERLAY
# ---------------------------------------------------------------------

COLOR_WHITE         = (245, 245, 245)
COLOR_DIM           = (175, 175, 175)
COLOR_PANEL         = (14, 14, 14)
COLOR_PANEL_EDGE    = (90, 90, 90)
COLOR_PANEL_EDGE_ACCENT = (120, 120, 120)  # lighter accent for panel inner edge
ROI_FILL            = (80, 100, 30)     # greenish, kept subtle (alpha 0.22)
ROI_EDGE            = (0, 200, 255)     # amber
LINE_COLOR          = (0, 200, 255)     # amber

# ── CCTV-style camera overlay ─────────────────────────────────────────────
# The camera view has to read as an airport CCTV feed, not as a computer
# vision demonstration. The queue area, the counting line and the tracked
# people are all still drawn, because the operator needs them, but every
# element is thin, matte and quiet: a faint wash instead of a filled block,
# a dashed boundary instead of a solid outline, one muted colour for the
# people instead of a rainbow, and no confidence scores.
CCTV_ROI_FILL       = (46, 58, 18)      # faint amber wash
CCTV_ROI_EDGE       = (0, 170, 210)     # thin amber boundary
CCTV_CAPTION        = (0, 215, 255)     # brighter amber for captions, so the
                                         # text stays readable on a light scene
CCTV_LINE           = (0, 170, 210)     # thin amber counting line
CCTV_TRACK          = (170, 190, 80)    # muted teal for tracked people
CCTV_TRACK_OUTSIDE  = (140, 140, 140)   # people outside the queue area
CCTV_SHADOW         = (18, 18, 18)      # one-pixel drop shadow for captions

# Enhanced color palette for improved visual hierarchy
COLOR_TEXT_PRIMARY   = (255, 255, 255)
COLOR_TEXT_SECONDARY = (200, 200, 200)
COLOR_TEXT_MUTED     = (150, 150, 150)
COLOR_ACCENT_GREEN   = (0, 230, 118)
COLOR_ACCENT_BLUE    = (0, 200, 255)
COLOR_ACCENT_ORANGE  = (0, 140, 255)
COLOR_ACCENT_RED     = (0, 40, 255)
COLOR_ACCENT_YELLOW  = (0, 215, 215)
COLOR_SUCCESS        = (0, 230, 118)
COLOR_WARNING        = (0, 215, 215)
COLOR_DANGER         = (0, 40, 255)
COLOR_INFO           = (0, 200, 255)

# Per-ID box colours (BGR) — vivid and distinct so each tracked person
# keeps a consistent colour while crossing the queue.
ID_COLORS = [
    (0, 230, 118),   # green
    (230, 118, 0),   # orange
    (0, 120, 230),   # blue
    (230, 0, 120),   # magenta
    (118, 230, 0),   # yellow-green
    (0, 230, 230),   # cyan
    (230, 230, 0),   # yellow
    (230, 0, 0),     # red
]


def _scale_coords(points, scale):
    """Map a list of native [x, y] coordinates into display space."""
    if scale == 1.0:
        return points
    return [[float(x) * scale, float(y) * scale] for x, y in points]


def _dashed_segment(out, p0, p1, color, thickness=1, dash=8, gap=6):
    """Draw one dashed line segment. OpenCV has no native dash support."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        cv2.line(
            out,
            (int(round(x0 + ux * pos)), int(round(y0 + uy * pos))),
            (int(round(x0 + ux * end)), int(round(y0 + uy * end))),
            color, thickness, cv2.LINE_AA,
        )
        pos = end + gap


def _dashed_polyline(out, pts, color, thickness=1, dash=8, gap=6):
    """Dashed outline of a polygon."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    for i in range(len(p)):
        _dashed_segment(out, p[i], p[(i + 1) % len(p)], color, thickness, dash, gap)


def _cctv_caption(out, text, org, scale, color):
    """A caption with a one-pixel shadow, the way CCTV overlays read."""
    x, y = int(org[0]), int(org[1])
    cv2.putText(out, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, CCTV_SHADOW, 2, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _draw_layout(frame, config, display_scale=1.0):
    """ROI polygon (faint wash, dashed boundary, quiet caption) + counting line.

    ``display_scale`` maps the native-pixel layout (ROI, counting line,
    X-ray side point) into the display frame, so the layout can be drawn
    directly on a downscaled copy of the capture.

    Geometry source: the active mapping (native pixels) when one is
    loaded, else the legacy ``config.roi`` / ``config.counting_line``.
    Drawing and analytics always resolve through the same helper so a
    technician calibrating a mapping sees exactly the geometry the
    engine tests people with.
    """
    out = frame.copy()
    s = _ui_scale(out.shape[1])
    frame_w, frame_h = int(out.shape[1]), int(out.shape[0])

    # ── resolve geometry (mapping -> legacy fallback) ────────
    roi_pts = None
    line_a = None
    line_b = None
    if config.mapping_active and _MAPPING_AVAILABLE:
        try:
            pix = mapping_to_pixel(config.mapping, (frame_w, frame_h))
            roi_pts = pix.roi
            if pix.out_line is not None:
                line_a, line_b = pix.out_line[0], pix.out_line[1]
            elif pix.in_line is not None:
                line_a, line_b = pix.in_line[0], pix.in_line[1]
        except MappingError:
            roi_pts = None

    if roi_pts is None:
        if config.roi_configured:
            roi_pts = tuple((float(x), float(y)) for x, y in config.roi)
        if config.line_configured:
            line_a = (float(config.counting_line[0][0]), float(config.counting_line[0][1]))
            line_b = (float(config.counting_line[1][0]), float(config.counting_line[1][1]))

    s_draw = float(display_scale)
    if roi_pts:
        roi = _scale_coords(roi_pts, s_draw)
        pts = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], CCTV_ROI_FILL)
        # A whisper of colour so the queue area is legible without becoming
        # a block that competes with the people in the frame.
        cv2.addWeighted(overlay, 0.07, out, 0.93, 0, out)
        _dashed_polyline(out, pts, CCTV_ROI_EDGE, 1, 7 * s, 5 * s)
        _roi_badge(out, roi, s)

    if line_a is not None and line_b is not None:
        ax = (round(float(line_a[0]) * s_draw, 4), round(float(line_a[1]) * s_draw, 4))
        bx = (round(float(line_b[0]) * s_draw, 4), round(float(line_b[1]) * s_draw, 4))
        # A single thin dashed line. The old overlay drew a black and a
        # white line under it plus two large badges, which read as a
        # vision demo rather than as a checkpoint.
        _dashed_segment(out, ax, bx, CCTV_LINE, 1, 9 * s, 7 * s)
        # Short end ticks mark the counting line without labelling it.
        tick = int(10 * s)
        for (px, py), (qx, qy) in ((ax, bx), (bx, ax)):
            dx, dy = qx - px, qy - py
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            cv2.line(out,
                     (int(px - nx * tick), int(py - ny * tick)),
                     (int(px + nx * tick), int(py + ny * tick)),
                     CCTV_LINE, 1, cv2.LINE_AA)
    return out


def _ui_scale(w):
    """HUD geometry/font scale so the layout stays proportional 640→1920 px."""
    return max(0.55, min(2.0, w / 1280.0))


def _text(out, text, org, scale, color, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(out, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _text_right(out, text, right_x, y, scale, color, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    _text(out, text, (right_x - tw, y), scale, color, thickness)


def _panel(out, x0, y0, x1, y1, alpha=0.82, fill=COLOR_PANEL, edge=COLOR_PANEL_EDGE, edge_accent=None):
    """Translucent dark panel with a sharp 1px border, clipped to the frame.
    
    If edge_accent is provided, draws a subtle inner highlight for depth.
    """
    h, w = out.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    cv2.rectangle(out, (x0, y0), (x1, y1), edge, 1)
    if edge_accent:
        cv2.rectangle(out, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), edge_accent, 1)


def _badge(out, text, center, scale, color, thickness=2):
    """Text on a black pill with a coloured border (readable over any video)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x0, y0 = int(center[0] - tw / 2) - 6, int(center[1] - th / 2) - 6
    x1, y1 = int(center[0] + tw / 2) + 6, int(center[1] + th / 2) + 7
    cv2.rectangle(out, (x0, y0), (x1, y1), (10, 10, 10), cv2.FILLED)
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
    _text(out, text, (x0 + 6, y0 + th + 1), scale, color, thickness)


def _card_panel(out, x0, y0, x1, y1, s, fill=COLOR_PANEL, edge=COLOR_PANEL_EDGE):
    """Enhanced card-style panel with subtle gradient effect and double border."""
    h, w = out.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, cv2.FILLED)
    cv2.addWeighted(overlay, 0.92, out, 0.08, 0, out)
    cv2.rectangle(out, (x0, y0), (x1, y1), edge, 1)
    # Subtle inner highlight for depth
    cv2.line(out, (x0 + 1, y0 + 1), (x1 - 1, y0 + 1), (50, 50, 50), 1)
    cv2.line(out, (x0 + 1, y0 + 1), (x0 + 1, y1 - 1), (50, 50, 50), 1)


def _roi_badge(out, roi, s):
    """A quiet 'QUEUE AREA' caption just inside the top of the ROI.

    It sits inside the boundary rather than above it, because the space above
    the queue area belongs to the frame's own camera overlay.
    """
    pts = np.array(roi, dtype=np.float32).reshape(-1, 2)
    top = pts[int(np.argmin(pts[:, 1]))]
    _cctv_caption(out, "QUEUE AREA", (int(top[0]) + int(6 * s), int(top[1]) + int(16 * s)), 0.42 * s, CCTV_CAPTION)


def _box_label(out, text, org, scale, color, bg=(10, 10, 10)):
    """Small filled strip behind a label so text stays sharp on video."""
    x, y = int(org[0]), int(org[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y), bg, cv2.FILLED)
    cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y), color, 1)
    _text(out, text, (x + 3, y - 4), scale, color, 2)


def _metric_card(out, x, y, label, value, label_color, value_color, s, width=None):
    """Draw a metric card with label and value, returns the next y position."""
    lw = int(12 * s)
    _text(out, label, (x, y), 0.38 * s, label_color, 1)
    _text(out, value, (x, y + int(16 * s)), 0.48 * s, value_color, 2)
    return y + int(42 * s)


def _separator(out, x0, y, x1, s, color=COLOR_PANEL_EDGE):
    """Draw a horizontal separator line."""
    cv2.line(out, (int(x0), int(y)), (int(x1), int(y)), color, 1)


def _pill(out, x, y, text, scale, color, s, padding=8):
    """Draw a pill-shaped tag with text."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    px = int(x)
    py = int(y)
    pad = int(padding * s)
    cv2.rectangle(out, (px - pad, py - th - pad), (px + tw + pad, py + pad), (10, 10, 10), cv2.FILLED)
    cv2.rectangle(out, (px - pad, py - th - pad), (px + tw + pad, py + pad), color, 1)
    _text(out, text, (px, py), scale, color, 2)
    return tw + 2 * pad


def _occ_color(ratio, config):
    r = float(ratio)
    if r >= config.occupancy_congested:
        return _STATE_COLORS[CONGESTED]
    if r >= config.occupancy_crowded:
        return _STATE_COLORS[CROWDED]
    if r >= config.occupancy_normal:
        return _STATE_COLORS[NORMAL]
    if r >= config.occupancy_sparse:
        return _STATE_COLORS[SPARSE]
    return _STATE_COLORS[EMPTY]


def _metrics(snapshot, config, show_raw_density=False):
    state = snapshot.get("state", UNCONFIGURED)
    flow = snapshot.get("flow", {}) or {}
    waiting = snapshot.get("waiting", {}) or {}
    spacing = snapshot.get("spacing", {}) or {}
    cong = snapshot.get("congestion", {}) or {}
    backlog = snapshot.get("backlog", {}) or {}
    roi_ok = (snapshot.get("config_status", {}) or {}).get("roi_configured", config.roi_configured)

    occ_pct = snapshot.get("queue_occupancy_percent")
    occ_ratio = (occ_pct / 100.0) if (roi_ok and occ_pct is not None) else None
    occ_text = f"{occ_pct:.0f}%" if occ_ratio is not None else "UNAVAILABLE"
    occ_color = _occ_color(occ_ratio, config) if occ_ratio is not None else _STATE_COLORS[UNCONFIGURED]

    flow_cur = flow.get("current_per_min")
    flow_avg = flow.get("avg_5min_per_min")
    flow_text = f"{flow_cur:.1f}/min" if flow_cur is not None else "n/a"
    if flow_avg is not None:
        flow_text += f"  {flow_avg:.1f} avg"

    avg_wait = waiting.get("average_s", 0)
    people = snapshot.get("people_in_queue", 0)
    wait_text = fmt_duration(avg_wait) if (avg_wait or people) else "n/a"

    sp = spacing.get("average_px")
    spacing_text = f"{sp:.0f} px" if sp is not None else "n/a"

    dur = cong.get("duration_s", 0)
    if cong.get("active"):
        cong_text, cong_color = "ACTIVE " + fmt_duration(dur), _STATE_COLORS[CONGESTED]
    else:
        cong_text, cong_color = "none", COLOR_DIM

    if backlog.get("check") == "UNAVAILABLE":
        backlog_text, backlog_color = "UNAVAILABLE", COLOR_DIM
    elif state == BACKLOG:
        backlog_text, backlog_color = "ACTIVE", _STATE_COLORS[BACKLOG]
    else:
        backlog_text, backlog_color = "OK", (0, 230, 118)

    return {
        "status": state.replace("_", " "),
        "status_color": _STATE_COLORS.get(state, _STATE_COLORS[UNCONFIGURED]),
        "people": str(people),
        "flow": flow_text,
        "wait": wait_text,
        "spacing": spacing_text,
        "occupancy": occ_text,
        "occupancy_color": occ_color,
        "occ_ratio": occ_ratio,
        "congestion": cong_text,
        "congestion_color": cong_color,
        "backlog": backlog_text,
        "backlog_color": backlog_color,
        "calibration": "METRES" if config.calibrated else "PX ONLY",
    }


def _draw_metrics_row(out, y, s, cells, bars=None):
    bars = bars or {}
    h, w = out.shape[:2]
    bar_h = int(56 * s)
    _panel(out, 0, y, w, y + bar_h, alpha=0.88, edge_accent=COLOR_PANEL_EDGE_ACCENT)
    n = len(cells)
    cell_w = w / n
    for i, (label, value, color) in enumerate(cells):
        x0 = int(i * cell_w) + int(12 * s)
        _text(out, label, (x0, y + int(18 * s)), 0.4 * s, COLOR_TEXT_MUTED, 1)
        _text(out, value, (x0, y + int(44 * s)), 0.6 * s, color, 2)
        if i in bars and bars[i] is not None:
            r = max(0.0, min(1.0, float(bars[i])))
            bw = int(cell_w) - int(24 * s)
            cv2.rectangle(out, (x0, y + int(27 * s)), (x0 + bw, y + int(33 * s)), (40, 40, 40), cv2.FILLED)
            cv2.rectangle(out, (x0, y + int(27 * s)), (x0 + int(bw * r), y + int(33 * s)), COLOR_SUCCESS, cv2.FILLED)


def _draw_header(out, snapshot, s, frame_index=None):
    h, w = out.shape[:2]
    bar_h = int(70 * s)
    _panel(out, 0, 0, w, bar_h, alpha=0.90, edge_accent=COLOR_PANEL_EDGE_ACCENT)
    state = snapshot.get("state", UNCONFIGURED)
    col = _STATE_COLORS.get(state, _STATE_COLORS[UNCONFIGURED])
    state_label = state.replace("_", " ")

    _text(out, "AIRPORT X-RAY QUEUE MONITOR", (int(14 * s), int(26 * s)),
          0.62 * s, COLOR_TEXT_PRIMARY, 2, cv2.FONT_HERSHEY_DUPLEX)
    _text(out, "CAMERA-ONLY PROTOTYPE", (int(15 * s), int(49 * s)), 0.42 * s, COLOR_INFO, 1)

    # LIVE indicator + running frame number, right after the subtitle.
    # The frame counter must increment every processed frame — visible
    # proof the UI is refreshing.
    (sub_w, _), _ = cv2.getTextSize("CAMERA-ONLY PROTOTYPE", cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, 1)
    live_x = int(15 * s) + sub_w + int(22 * s)
    cv2.circle(out, (live_x, int(41 * s)), int(7 * s), (0, 80, 40), -1, cv2.LINE_AA)
    cv2.circle(out, (live_x, int(41 * s)), int(5 * s), COLOR_SUCCESS, -1, cv2.LINE_AA)
    _text(out, "LIVE", (live_x + int(12 * s), int(49 * s)), 0.42 * s, COLOR_SUCCESS, 1)
    if frame_index is not None:
        _text(out, f"FRAME {frame_index:06d}", (live_x + int(72 * s), int(49 * s)), 0.42 * s, COLOR_TEXT_SECONDARY, 1)

    # state chip (top-right, left of the clock)
    (tw, th), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, 2)
    chip_w = tw + int(24 * s)
    chip_h = int(30 * s)
    cx0 = w - int(252 * s) - chip_w
    cy0 = int(14 * s)
    cv2.rectangle(out, (cx0, cy0), (cx0 + chip_w, cy0 + chip_h), col, cv2.FILLED)
    cv2.rectangle(out, (cx0, cy0), (cx0 + chip_w, cy0 + chip_h), (60, 60, 60), 1)
    cv2.line(out, (cx0 + 1, cy0 + 1), (cx0 + chip_w - 1, cy0 + 1), (80, 80, 80), 1)
    _text(out, state_label, (cx0 + int(12 * s), cy0 + int(22 * s)), 0.55 * s, (10, 10, 10), 2)

    # clock: full local date + time, right-aligned
    now = datetime.now()
    right = w - int(14 * s)
    _text_right(out, now.strftime("%Y-%m-%d"), right, int(24 * s), 0.5 * s, COLOR_TEXT_MUTED, 1)
    _text_right(out, now.strftime("%H:%M:%S"), right, int(52 * s), 0.7 * s, COLOR_TEXT_PRIMARY, 2)


def _draw_video_layer(frame, tracks, config, track_confs=None, display_scale=1.0,
                      show_console_tip=True):
    """Camera view layer: layout geometry + tracked people.

    Pure overlay on the (already downscaled) video frame - no HUD panels.
    Used by both the legacy ``draw_overlay`` and the dashboard viewport.

    ``show_console_tip`` adds the keyboard hint that only makes sense in the
    standalone desktop window. The browser dashboard omits it.

    ``track_confs`` is accepted for call-site compatibility. Confidence
    scores are no longer drawn: the camera view is an operational feed, not
    a detector readout.
    """
    out = _draw_layout(frame, config, display_scale=display_scale)
    s = _ui_scale(out.shape[1])
    h, w = out.shape[:2]
    k = display_scale

    # Resolve the geometry the track colouring tests against: the active
    # mapping in native pixels, or the legacy ROI. Mirrors the analytics
    # engine so a technician calibrating sees the same membership rules.
    roi_pts = None
    ref_mode = CENTROID_REFERENCE
    if config.mapping_active and _MAPPING_AVAILABLE:
        try:
            pix = mapping_to_pixel(config.mapping, (int(frame.shape[1]), int(frame.shape[0])))
            roi_pts = tuple(pix.roi)
            ref_mode = pix.reference_point or CENTROID_REFERENCE
        except MappingError:
            roi_pts = None
    if roi_pts is None and config.roi:
        roi_pts = tuple((float(x), float(y)) for x, y in config.roi)

    for t in tracks:
        x1, y1, x2, y2 = int(t[0] * k), int(t[1] * k), int(t[2] * k), int(t[3] * k)
        tid = int(t[4])
        ncx, ncy = reference_xy((t[0], t[1], t[2], t[3]), ref_mode)
        inside = _in_polygon(ncx, ncy, roi_pts) if roi_pts else False
        color = CCTV_TRACK if inside else CCTV_TRACK_OUTSIDE
        # A thin outline, one muted colour. No corner brackets, no centroid
        # marker and no confidence score: the operator needs to know where
        # people are, not how sure the detector is.
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        # An anonymous tracking id, small enough to stay out of the way.
        _cctv_caption(out, str(tid), (x1, y1 - 3), 0.38 * s, color)

    if not config.roi_configured:
        _panel(out, 0, 0, w, int(76 * s), alpha=0.85)
        _text(out, "QUEUE AREA NOT CONFIGURED", (int(12 * s), int(30 * s)), 0.7 * s, (0, 40, 255), 2)
        if show_console_tip:
            _text(out, "TIP: press 'r' to reload config/camera.yaml", (int(12 * s), int(58 * s)), 0.5 * s, (220, 220, 220), 1)
    return out


def draw_overlay(frame, snapshot, tracks, config, fps=0.0, show_raw_density=False,
                 track_confs=None, inf_ms=None, debug=False, frame_index=None,
                 display_scale=1.0, disp_ms=None):
    """Full camera-only overlay: boxes + ROI + line + header/footer HUD.

    ``display_scale`` maps native-pixel coordinates (ROI, counting line,
    track boxes, centroids) into the display frame, so the HUD is drawn
    directly on a downscaled copy of the capture instead of on the full
    native frame followed by a resize. Analytics stay in native pixels.
    """
    out = _draw_video_layer(frame, tracks, config,
                            track_confs=track_confs, display_scale=display_scale)
    s = _ui_scale(out.shape[1])
    h, w = out.shape[:2]

    _draw_header(out, snapshot, s, frame_index=frame_index)

    m = _metrics(snapshot, config, show_raw_density)
    y2 = h - int(112 * s)
    _draw_metrics_row(out, y2, s, [
        ("STATUS", m["status"], m["status_color"]),
        ("PEOPLE", m["people"], COLOR_WHITE),
        ("FLOW", m["flow"], COLOR_WHITE),
        ("AVG WAIT", m["wait"], COLOR_WHITE),
        ("SPACING", m["spacing"], COLOR_WHITE),
    ])
    y1 = y2 - int(56 * s)
    _draw_metrics_row(out, y1, s, [
        ("QUEUE OCCUPANCY", m["occupancy"], m["occupancy_color"]),
        ("CONGESTION", m["congestion"], m["congestion_color"]),
        ("BACKLOG", m["backlog"], m["backlog_color"]),
        ("CALIBRATION", m["calibration"], COLOR_DIM),
    ], bars={0: m["occ_ratio"]})

    # FPS / inference corner under the header
    header_h = int(70 * s)
    corner = f"FPS {fps:5.1f}" if fps > 0 else "FPS  —"
    if inf_ms is not None:
        corner += f"   inf {inf_ms:4.1f} ms"
    if disp_ms is not None:
        corner += f"   disp {disp_ms:4.1f} ms"
    _text(out, corner, (int(12 * s), header_h + int(22 * s)), 0.5 * s, COLOR_DIM, 1)

    if debug:
        reasons = snapshot.get("reasons", [])
        ry = header_h + int(44 * s)
        _text(out, "REASONS:", (int(12 * s), ry), 0.5 * s, (210, 210, 210), 1)
        for i, r in enumerate(reasons[:5]):
            _text(out, f"  - {r}", (int(12 * s), ry + int(20 * s) * (i + 1)), 0.45 * s, (210, 210, 210), 1)

    return out


def preview_layout(frame, config):
    """Static preview of a saved layout (ROI + line + direction)."""
    out = _draw_layout(frame, config)
    s = _ui_scale(out.shape[1])
    notes = ["LAYOUT PREVIEW  (edit config/camera.yaml or re-run --define-roi)"]
    if not config.roi_configured:
        notes.append("ROI: NOT CONFIGURED")
    if not config.line_configured:
        notes.append("Counting line: NOT CONFIGURED")
    _panel(out, 0, 0, int(560 * s), int(64 * s), alpha=0.85)
    _text(out, " - ".join(notes), (int(12 * s), int(28 * s)), 0.55 * s, (0, 200, 255), 2)
    _text(out, "Press any key to close", (int(12 * s), int(52 * s)), 0.5 * s, COLOR_DIM, 1)
    return out


# ---------------------------------------------------------------------
#  DASHBOARD — SHARED METRICS MODEL
#  One model built per frame, consumed by BOTH the terminal console line
#  and the on-screen dashboard, so the two can never disagree.
# ---------------------------------------------------------------------


@dataclass
class SystemMetrics:
    """Measured pipeline performance (never faked)."""

    fps: float = 0.0
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    analytics_ms: float = 0.0
    render_ms: float = 0.0
    capture_fps: float = 0.0
    frame_count: int = 0
    detections: int = 0
    tracks: int = 0
    source: str = ""
    device: str = ""
    img_size: int = 0
    #: Epoch seconds of the newest captured frame (0 = not measured yet).
    #: Used to report the real frame age instead of assuming one interval.
    last_frame_unix: float = 0.0
    # ── Camera identity/orientation actually running (additive) ─────
    #: The *running* source, which a CLI --source override can differ
    #: from camera.yaml's persisted value.
    source_index: Optional[int] = None
    #: Operator-facing camera chrome label (e.g. "CAM-01").
    source_label: str = ""
    #: Clockwise rotation currently applied at the capture edge.
    rotation_deg: int = 0
    #: Actual frame size being analysed (post-rotation when rotation ≠ 0;
    #: 0 = not measured yet). Snapshot resolution reporting prefers these
    #: measured values over the requested config width/height.
    frame_w: int = 0
    frame_h: int = 0
    #: Camera lifecycle state reported by the capture supervisor:
    #: "OK" (delivering frames), "RECONNECTING" (camera lost, bounded
    #: retries in progress), "NO_FRAMES" (source opened but no frames
    #: delivered), "OFFLINE" (retries exhausted, awaiting manual retry),
    #: "" (unknown / older pipeline). Never fabricated — empty when the
    #: pipeline does not report it.
    camera_status: str = ""


@dataclass
class QueueMetrics:
    """Queue analytics presented on both the console and the dashboard."""

    state: str = UNCONFIGURED
    reasons: list = field(default_factory=list)
    state_duration_s: float = 0.0
    people_detected: int = 0
    people_in_queue: int = 0
    occupancy_percent: Optional[float] = None
    occupancy_avg_percent: Optional[float] = None
    entry_flow: float = 0.0
    exit_flow: float = 0.0
    total_entries: int = 0
    total_exits: int = 0
    avg_wait_s: float = 0.0
    longest_wait_s: float = 0.0
    predicted_wait_s: float = 0.0
    clearance_time_s: float = 0.0
    service_time_s: float = 0.0
    pacing_status: str = "STEADY"
    movement: str = "IDLE"
    spacing_px: Optional[float] = None
    congestion_active: bool = False
    congestion_duration_s: float = 0.0
    calibration: str = "PX ONLY"
    recommendation: str = ""
    config_status: dict = field(default_factory=dict)
    events: list = field(default_factory=list)


@dataclass
class DashboardModel:
    queue: QueueMetrics
    system: SystemMetrics


def build_dashboard_model(snapshot, system, config, recent_events=None) -> DashboardModel:
    """Derive the shared dashboard model from one analytics snapshot.

    ``recent_events`` is the rolling event timeline (list of event dicts,
    newest last) maintained by the caller.
    """
    flow = snapshot.get("flow", {}) or {}
    waiting = snapshot.get("waiting", {}) or {}
    spacing = snapshot.get("spacing", {}) or {}
    cong = snapshot.get("congestion", {}) or {}
    cs = snapshot.get("config_status", {}) or {}
    occ_avg = snapshot.get("occupancy_avg", {}) or {}
    roi_ok = cs.get("roi_configured", config.roi_configured)
    occ_pct = snapshot.get("queue_occupancy_percent")
    state = snapshot.get("state", UNCONFIGURED)

    queue = QueueMetrics(
        state=state,
        reasons=snapshot.get("reasons", []),
        state_duration_s=snapshot.get("state_duration_s", 0.0),
        people_detected=snapshot.get("people_detected", 0),
        people_in_queue=snapshot.get("people_in_queue", 0),
        occupancy_percent=occ_pct if roi_ok else None,
        occupancy_avg_percent=occ_avg.get("percent") if roi_ok else None,
        entry_flow=float(flow.get("entry_current_per_min", 0) or 0),
        exit_flow=float(flow.get("current_per_min", 0) or 0),
        total_entries=snapshot.get("total_entries", 0),
        total_exits=snapshot.get("total_exits", 0),
        avg_wait_s=float(waiting.get("average_s", 0) or 0),
        longest_wait_s=float(waiting.get("longest_s", 0) or 0),
        predicted_wait_s=float(waiting.get("predicted_s", 0) or 0),
        clearance_time_s=float(waiting.get("clearance_s", 0) or 0),
        service_time_s=float(snapshot.get("service_time_s", 0) or 0),
        pacing_status=str(snapshot.get("pacing", "STEADY")),
        movement=snapshot.get("movement", "IDLE"),
        spacing_px=spacing.get("average_px"),
        congestion_active=bool(cong.get("active")),
        congestion_duration_s=float(cong.get("duration_s", 0) or 0),
        calibration="METRES" if config.calibrated else "PX ONLY",
        recommendation=recommendation_for(state, config),
        config_status=cs,
        events=list(recent_events or []),
    )
    return DashboardModel(queue=queue, system=system)


class SeriesBuffer:
    """Rolling (x, y) history for a live mini-chart."""

    def __init__(self, history=120):
        self.data = deque(maxlen=max(2, int(history)))

    def append(self, x, y):
        self.data.append((float(x), float(y)))

    def values(self):
        return list(self.data)


class DashboardSeries:
    """Rolling buffers for the bottom trend strip."""

    def __init__(self, history=120):
        self.history = int(history)
        self.entry_flow = SeriesBuffer(self.history)
        self.exit_flow = SeriesBuffer(self.history)
        self.queue_size = SeriesBuffer(self.history)
        self.wait_time = SeriesBuffer(self.history)

    def append(self, frame_index, queue: QueueMetrics):
        self.entry_flow.append(frame_index, queue.entry_flow)
        self.exit_flow.append(frame_index, queue.exit_flow)
        self.queue_size.append(frame_index, queue.people_in_queue)
        self.wait_time.append(frame_index, queue.avg_wait_s)


# ---------------------------------------------------------------------
#  DASHBOARD — ON-SCREEN RENDERING
# ---------------------------------------------------------------------


def _wrap_text(text, max_px, scale, thickness=1) -> list:
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _event_time_local(iso_str) -> str:
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).astimezone()
        return dt.strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


def _event_label(ev) -> str:
    et = ev.get("event_type")
    if et == "line_crossing":
        return f"{ev.get('direction', '?')}  ID {ev.get('track_id', '?')}"
    if et == "congestion_start":
        return "CONGESTION START"
    if et == "congestion_end":
        return f"CONGESTION END {fmt_duration(ev.get('duration_s', 0))}"
    if et == "state_change":
        return f"STATE {ev.get('from_state', '?')} -> {ev.get('to_state', '?')}"
    if et == "track_lost":
        return f"TRACK LOST ID {ev.get('track_id', '?')}"
    if et == "track_enter_roi":
        return f"JOINED ID {ev.get('track_id', '?')}"
    return str(et).replace("_", " ").upper()


def _event_color(ev) -> tuple:
    et = ev.get("event_type")
    if et == "line_crossing":
        return (0, 230, 118) if ev.get("direction") == "ENTER" else (0, 180, 255)
    if et in ("congestion_start", "congestion_end"):
        return _STATE_COLORS[CONGESTED]
    if et == "state_change":
        return (230, 230, 230)
    if et == "track_lost":
        return COLOR_DIM
    return (0, 200, 255)


def _system_status(q: QueueMetrics) -> tuple:
    """(status_label, colour) — OPERATIONAL / CALIBRATION REQUIRED / UNCONFIGURED."""
    if not q.config_status.get("roi_configured"):
        return "UNCONFIGURED", _STATE_COLORS[UNCONFIGURED]
    if not q.config_status.get("calibrated"):
        return "CALIBRATION REQUIRED", (0, 215, 215)
    return "OPERATIONAL", (0, 230, 118)


def _draw_dashboard_header(canvas, model, config, s, frame_index=None):
    h, w = canvas.shape[:2]
    bar_h = int(64 * s)
    _panel(canvas, 0, 0, w, bar_h, alpha=0.95, edge_accent=COLOR_PANEL_EDGE_ACCENT)
    q = model.queue

    # Title block
    _text(canvas, "AIRPORT X-RAY QUEUE MONITOR", (int(14 * s), int(26 * s)),
          0.56 * s, COLOR_TEXT_PRIMARY, 2, cv2.FONT_HERSHEY_DUPLEX)
    _text(canvas, config.camera_id, (int(15 * s), int(50 * s)), 0.42 * s, COLOR_INFO, 1)

    # LIVE indicator with subtle ring
    (sub_w, _), _ = cv2.getTextSize("AIRPORT X-RAY QUEUE MONITOR", cv2.FONT_HERSHEY_DUPLEX, 0.56 * s, 2)
    live_x = int(15 * s) + sub_w + int(26 * s)
    cv2.circle(canvas, (live_x, int(42 * s)), int(7 * s), (0, 80, 40), -1, cv2.LINE_AA)  # outer ring
    cv2.circle(canvas, (live_x, int(42 * s)), int(5 * s), COLOR_SUCCESS, -1, cv2.LINE_AA)
    _text(canvas, "LIVE", (live_x + int(12 * s), int(50 * s)), 0.42 * s, COLOR_SUCCESS, 1)
    if frame_index is not None:
        _text(canvas, f"FRM {frame_index:06d}", (live_x + int(72 * s), int(50 * s)), 0.42 * s, COLOR_TEXT_SECONDARY, 1)

    # System status chip - enhanced with rounded corners simulation
    status, col = _system_status(q)
    (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, 2)
    chip_w = tw + int(24 * s)
    chip_h = int(30 * s)
    cx0 = w - int(150 * s) - chip_w
    cy0 = int(12 * s)
    # Draw chip background with subtle highlight
    cv2.rectangle(canvas, (cx0, cy0), (cx0 + chip_w, cy0 + chip_h), col, cv2.FILLED)
    cv2.rectangle(canvas, (cx0, cy0), (cx0 + chip_w, cy0 + chip_h), (60, 60, 60), 1)
    # Inner highlight
    cv2.line(canvas, (cx0 + 1, cy0 + 1), (cx0 + chip_w - 1, cy0 + 1), (80, 80, 80), 1)
    _text(canvas, status, (cx0 + int(12 * s), cy0 + int(22 * s)), 0.5 * s, (10, 10, 10), 2)

    # Clock area with date/time
    now = datetime.now()
    right = w - int(14 * s)
    _text_right(canvas, now.strftime("%Y-%m-%d"), right, int(24 * s), 0.5 * s, COLOR_TEXT_MUTED, 1)
    _text_right(canvas, now.strftime("%H:%M:%S"), right, int(52 * s), 0.7 * s, COLOR_TEXT_PRIMARY, 2)


def _draw_performance(canvas, model, s, top):
    """Compact pipeline-performance strip overlaid on the video viewport."""
    q = model.queue
    sysm = model.system
    h, w = canvas.shape[:2]
    panel_h = int(46 * s)
    _panel(canvas, int(6 * s), top + int(6 * s), int(460 * s), top + int(6 * s) + panel_h, alpha=0.80, edge_accent=COLOR_PANEL_EDGE_ACCENT)
    x = int(16 * s)
    y = top + int(20 * s)
    fps = f"FPS {sysm.fps:5.1f}" if sysm.fps > 0 else "FPS  --"
    _text(canvas, fps, (x, y), 0.42 * s, COLOR_TEXT_PRIMARY, 1)
    _text(canvas, f"INF {sysm.inference_ms:4.1f}ms", (x + int(86 * s), y), 0.42 * s, COLOR_TEXT_MUTED, 1)
    _text(canvas, f"TRK {sysm.tracking_ms:4.1f}ms", (x + int(176 * s), y), 0.42 * s, COLOR_TEXT_MUTED, 1)
    _text(canvas, f"ANL {sysm.analytics_ms:4.1f}ms", (x + int(266 * s), y), 0.42 * s, COLOR_TEXT_MUTED, 1)
    _text(canvas, f"RND {sysm.render_ms:4.1f}ms", (x + int(356 * s), y), 0.42 * s, COLOR_TEXT_MUTED, 1)
    y2 = top + int(42 * s)
    cap = f"CAP {sysm.capture_fps:4.1f}fps" if sysm.capture_fps > 0 else "CAP  --"
    _text(canvas, cap, (x, y2), 0.4 * s, COLOR_TEXT_MUTED, 1)
    _text(canvas, f"DET {sysm.detections}", (x + int(100 * s), y2), 0.4 * s, COLOR_TEXT_MUTED, 1)
    _text(canvas, f"TRK {sysm.tracks}", (x + int(180 * s), y2), 0.4 * s, COLOR_TEXT_MUTED, 1)
    _text(canvas, f"SRC {sysm.source}", (x + int(260 * s), y2), 0.4 * s, COLOR_TEXT_MUTED, 1)


def _draw_dashboard_panel(canvas, model, config, s, panel_w, top, height):
    q = model.queue
    h, w = canvas.shape[:2]
    px = w - panel_w
    _card_panel(canvas, px, top, w, top + height, s)
    x = px + int(12 * s)
    iw = panel_w - int(24 * s)
    half = iw // 2
    y = top + int(14 * s)
    bottom = top + height - int(6 * s)

    # state chip - enhanced
    col = _STATE_COLORS.get(q.state, _STATE_COLORS[UNCONFIGURED])
    label = q.state.replace("_", " ")
    chip_h = int(38 * s)
    cv2.rectangle(canvas, (x, y), (x + iw, y + chip_h), col, cv2.FILLED)
    cv2.rectangle(canvas, (x, y), (x + iw, y + chip_h), (60, 60, 60), 1)
    # Inner highlight
    cv2.line(canvas, (x + 1, y + 1), (x + iw - 1, y + 1), (80, 80, 80), 1)
    _text(canvas, f"QUEUE STATUS: {label}", (x + int(12 * s), y + int(27 * s)), 0.5 * s, (10, 10, 10), 2)
    y += chip_h + int(10 * s)

    # operational recommendation
    for line in _wrap_text(q.recommendation, iw, 0.42 * s, 1):
        if y + int(18 * s) > bottom:
            break
        _text(canvas, line, (x, y + int(15 * s)), 0.42 * s, COLOR_INFO, 1)
        y += int(17 * s)
    y += int(8 * s)

    # Separator
    _separator(canvas, x, y, x + iw, s, COLOR_PANEL_EDGE)
    y += int(8 * s)

    # metric grid - enhanced with new helper
    occ_ratio = (q.occupancy_percent / 100.0) if q.occupancy_percent is not None else None
    occ_txt = f"{q.occupancy_percent:.0f}%" if occ_ratio is not None else "UNAVAILABLE"
    occ_col = _occ_color(occ_ratio, config) if occ_ratio is not None else _STATE_COLORS[UNCONFIGURED]
    sp_txt = f"{q.spacing_px:.0f} px" if q.spacing_px is not None else "n/a"
    mov_col = {"MOVING": COLOR_SUCCESS, "STAGNANT": COLOR_WARNING, "IDLE": COLOR_TEXT_MUTED}.get(q.movement, COLOR_TEXT_MUTED)

    def grid_row(c1l, c1v, c1c, c2l, c2v, c2c):
        nonlocal y
        if y + int(46 * s) > bottom:
            return
        _text(canvas, c1l, (x, y), 0.38 * s, COLOR_TEXT_MUTED, 1)
        _text(canvas, c1v, (x, y + int(16 * s)), 0.48 * s, c1c, 2)
        _text(canvas, c2l, (x + half, y), 0.38 * s, COLOR_TEXT_MUTED, 1)
        _text(canvas, c2v, (x + half, y + int(16 * s)), 0.48 * s, c2c, 2)
        y += int(46 * s)

    grid_row("PEOPLE", str(q.people_in_queue), COLOR_TEXT_PRIMARY, "OCCUPANCY", occ_txt, occ_col)
    grid_row("ENTRY FLOW", f"{q.entry_flow:.1f}/min", COLOR_SUCCESS, "EXIT FLOW", f"{q.exit_flow:.1f}/min", COLOR_INFO)
    grid_row("ENTRIES", str(q.total_entries), COLOR_SUCCESS, "EXITS", str(q.total_exits), COLOR_INFO)
    grid_row("AVG WAIT", fmt_duration(q.avg_wait_s), COLOR_TEXT_PRIMARY, "MOVEMENT", q.movement, mov_col)
    grid_row("SPACING", sp_txt, COLOR_TEXT_PRIMARY, "CALIB.", q.calibration, COLOR_TEXT_MUTED)

    # congestion block - enhanced
    y += int(6 * s)
    if y + int(22 * s) > bottom:
        return
    _text(canvas, "CONGESTION", (x, y + int(15 * s)), 0.4 * s, COLOR_TEXT_MUTED, 1)
    if q.congestion_active:
        cong_txt = f"ACTIVE {fmt_duration(q.congestion_duration_s)}"
        cong_col = _STATE_COLORS[CONGESTED]
    else:
        cong_txt = "none"
        cong_col = COLOR_TEXT_MUTED
    _text_right(canvas, cong_txt, px + panel_w - int(12 * s), y + int(15 * s), 0.46 * s, cong_col, 2)
    y += int(28 * s)

    # Separator
    _separator(canvas, x, y, x + iw, s, COLOR_PANEL_EDGE)
    y += int(8 * s)

    # configuration checklist (UNCONFIGURED mode)
    if not q.config_status.get("roi_configured") or not q.config_status.get("line_configured"):
        if y + int(22 * s) > bottom:
            return
        _text(canvas, "CONFIGURATION", (x, y + int(15 * s)), 0.4 * s, COLOR_TEXT_MUTED, 1)
        y += int(24 * s)
        items = [
            ("Queue ROI polygon", q.config_status.get("roi_configured")),
            ("Counting line + X-ray side", q.config_status.get("line_configured")),
        ]
        for name, ok in items:
            if y + int(20 * s) > bottom:
                break
            mark = "[ OK ]" if ok else "[    ]"
            mcol = COLOR_SUCCESS if ok else COLOR_DANGER
            _text(canvas, f"{mark} {name}", (x, y + int(16 * s)), 0.42 * s, mcol if ok else COLOR_TEXT_PRIMARY, 1)
            y += int(20 * s)
        if y + int(20 * s) <= bottom:
            _text(canvas, "RUN: --define-roi", (x, y + int(16 * s)), 0.42 * s, COLOR_INFO, 1)
            y += int(20 * s)
        y += int(6 * s)

        # Separator
        _separator(canvas, x, y, x + iw, s, COLOR_PANEL_EDGE)
        y += int(8 * s)

    # event timeline
    if y + int(20 * s) > bottom:
        return
    _text(canvas, "EVENT TIMELINE", (x, y + int(15 * s)), 0.4 * s, COLOR_TEXT_MUTED, 1)
    y += int(24 * s)
    events = list(q.events)
    for ev in reversed(events[-8:]):
        if y + int(18 * s) > bottom:
            break
        t = _event_time_local(ev.get("event_time"))
        lbl = _event_label(ev)
        c = _event_color(ev)
        _text(canvas, t, (x, y + int(15 * s)), 0.38 * s, COLOR_TEXT_MUTED, 1)
        _text(canvas, lbl, (x + int(78 * s), y + int(15 * s)), 0.38 * s, c, 1)
        y += int(18 * s)
    if not events:
        _text(canvas, "No events yet", (x, y + int(15 * s)), 0.38 * s, COLOR_TEXT_MUTED, 1)


def _draw_chart(canvas, x0, y0, x1, y1, buf, label, color, fmt="%.1f"):
    h, w = canvas.shape[:2]
    x0, y0 = int(x0), int(y0)
    x1, y1 = min(int(x1), w - 1), min(int(y1), h - 1)
    if x1 <= x0 or y1 <= y0:
        return
    s = _ui_scale(w)
    _card_panel(canvas, x0, y0, x1, y1, s)
    data = buf.values()
    cur = data[-1][1] if data else 0.0
    # Draw horizontal grid lines
    grid_lines = 3
    for i in range(1, grid_lines):
        gy = y1 - 2 - int((i / grid_lines) * (y1 - y0 - 12))
        cv2.line(canvas, (x0 + int(4 * s), gy), (x1 - int(4 * s), gy), (40, 40, 40), 1)
    cv2.line(canvas, (x0, y1 - 2), (x1, y1 - 2), COLOR_PANEL_EDGE, 1)
    if len(data) >= 2:
        lo = min(v for _, v in data)
        hi = max(v for _, v in data)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        n = len(data)
        span_x = x1 - x0
        span_y = y1 - y0 - 12
        pts = []
        for i, (_, v) in enumerate(data):
            px = x0 + int((i / (n - 1)) * span_x)
            py = y1 - 2 - int(((v - lo) / (hi - lo)) * span_y)
            pts.append((px, py))
        # Draw area fill under the line
        area_pts = pts + [(x1, y1 - 2), (x0, y1 - 2)]
        cv2.fillPoly(canvas, [np.array(area_pts, dtype=np.int32)], (*color, 30))
        # Draw the line
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
        # Highlight current value point
        if pts:
            cv2.circle(canvas, pts[-1], int(4 * s), color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], int(6 * s), (10, 10, 10), 1, cv2.LINE_AA)
    _text(canvas, label, (x0 + int(8 * s), y0 + int(18 * s)), 0.42 * s, COLOR_TEXT_MUTED, 1)
    _text_right(canvas, fmt % cur, x1 - int(8 * s), y0 + int(18 * s), 0.42 * s, color, 1)


def _draw_dashboard_charts(canvas, model, series, s, top, chart_h):
    h, w = canvas.shape[:2]
    _card_panel(canvas, 0, top, w, top + chart_h, s)
    n = 4
    pad = int(10 * s)
    gap = int(8 * s)
    cw = (w - 2 * pad - (n - 1) * gap) / n
    charts = [
        (series.entry_flow, "ENTRY FLOW /min", COLOR_SUCCESS, "%.1f"),
        (series.exit_flow, "EXIT FLOW /min", COLOR_INFO, "%.1f"),
        (series.queue_size, "QUEUE SIZE", COLOR_WARNING, "%.0f"),
        (series.wait_time, "AVG WAIT /s", COLOR_ACCENT_BLUE, "%.0f"),
    ]
    for i, (buf, label, color, fmt) in enumerate(charts):
        x0 = pad + i * (cw + gap)
        _draw_chart(canvas, x0, top + int(8 * s), x0 + cw, top + chart_h - int(8 * s), buf, label, color, fmt)


def draw_dashboard(video_frame, model, config, tracks, track_confs=None,
                   frame_index=None, series=None, display_scale=1.0):
    """Compose the professional monitoring dashboard canvas.

    Layout: header bar (top) · camera viewport + right status panel (middle)
    · live trend charts (bottom). ``model`` is the shared DashboardModel.
    """
    video_h, video_w = video_frame.shape[:2]
    panel_w = max(200, int(config.dashboard_panel_width * _ui_scale(video_w + config.dashboard_panel_width)))
    s = _ui_scale(video_w + panel_w)
    header_h = int(64 * s)
    chart_h = int(160 * s)
    W = video_w + panel_w
    H = header_h + video_h + chart_h
    canvas = np.full((H, W, 3), 14, dtype=np.uint8)

    view = _draw_video_layer(video_frame, tracks, config,
                             track_confs=track_confs, display_scale=display_scale)
    canvas[header_h:header_h + video_h, 0:video_w] = view

    _draw_dashboard_header(canvas, model, config, s, frame_index=frame_index)
    _draw_performance(canvas, model, s, top=header_h)
    _draw_dashboard_panel(canvas, model, config, s, panel_w, header_h, video_h)
    if series is not None:
        _draw_dashboard_charts(canvas, model, series, s, header_h + video_h, chart_h)
    return canvas


# ---------------------------------------------------------------------
#  INTERACTIVE LAYOUT TOOL (ROI polygon + counting line + X-ray side)
# ---------------------------------------------------------------------


class _LayoutState:
    def __init__(self, shape):
        self.h, self.w = shape
        self.roi = []
        self.line = []
        self.side = None
        self.step = 0  # 0=roi, 1=line, 2=xray side point

    @property
    def step_label(self):
        return ["STEP 1/3 - click ROI polygon points (>=3), Enter to finish",
                "STEP 2/3 - click 2 counting-line endpoints, Enter to finish",
                "STEP 3/3 - click the X-RAY side point, then Enter to save"][self.step]


def _draw_layout_state(img, st: _LayoutState):
    out = img.copy()
    if st.roi:
        pts = np.array(st.roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 200, 255), 2)
        for p in st.roi:
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 200, 255), -1)
    if len(st.line) == 1:
        cv2.circle(out, (int(st.line[0][0]), int(st.line[0][1])), 5, (0, 140, 255), -1)
    if len(st.line) == 2:
        cv2.line(out, (int(st.line[0][0]), int(st.line[0][1])), (int(st.line[1][0]), int(st.line[1][1])),
                 (0, 140, 255), 3)
    if st.side:
        cv2.circle(out, (int(st.side[0]), int(st.side[1])), 7, (255, 0, 200), -1)
        cv2.putText(out, "X-RAY SIDE", (int(st.side[0]) + 10, int(st.side[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(out, st.step_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Keys: r=reset step  b=back  s=save  Esc=cancel", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def _layout_errors(st: _LayoutState) -> list:
    errors = []
    if len(st.roi) < 3:
        errors.append("ROI needs at least 3 polygon points")
    elif _polygon_area_px(st.roi) <= 0:
        errors.append("ROI polygon is degenerate (zero area)")
    if len(st.line) != 2:
        errors.append("counting line needs exactly 2 endpoints")
    if st.side is None:
        errors.append("X-ray side point is required")
    for name, pts in (("ROI", st.roi), ("line", st.line)):
        for x, y in pts:
            if not (0 <= x <= st.w and 0 <= y <= st.h):
                errors.append(f"{name} point ({x}, {y}) is outside the frame")
    if st.side and not (0 <= st.side[0] <= st.w and 0 <= st.side[1] <= st.h):
        errors.append("X-ray side point is outside the frame")
    return errors


def _save_layout(config: CameraConfig, st: _LayoutState, path):
    config.roi = [[float(x), float(y)] for x, y in st.roi]
    config.counting_line = [[float(x), float(y)] for x, y in st.line]
    config.xray_side_point = [float(st.side[0]), float(st.side[1])]
    config.source_shape_w = st.w
    config.source_shape_h = st.h
    config.save(path)
    return config


def define_layout_interactive(frame, config_path) -> bool:
    """Click-to-draw ROI polygon, counting line, and X-ray side point.

    Returns True if a layout was saved to ``config_path``.
    """
    window = "Define Queue Layout"
    cv2.namedWindow(window)
    st = _LayoutState(frame.shape[:2])

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if st.step == 0:
                st.roi.append((x, y))
            elif st.step == 1:
                if len(st.line) < 2:
                    st.line.append((x, y))
            else:
                st.side = (x, y)

    cv2.setMouseCallback(window, on_mouse)
    base = frame.copy()
    saved = False
    try:
        while True:
            img = _draw_layout_state(base, st)
            cv2.imshow(window, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if st.step == 0 and len(st.roi) >= 3:
                    st.step = 1
                elif st.step == 1 and len(st.line) == 2:
                    st.step = 2
                elif st.step == 2 and st.side is not None:
                    errors = _layout_errors(st)
                    if errors:
                        print("[ROI] Cannot save:")
                        for e in errors:
                            print(f"       - {e}")
                    else:
                        _save_layout(load_camera_config(config_path), st, config_path)
                        saved = True
                        print(f"[ROI] Layout saved to {config_path}")
                        break
            elif key == ord("r"):
                if st.step == 0:
                    st.roi = []
                elif st.step == 1:
                    st.line = []
                else:
                    st.side = None
            elif key == ord("b"):
                st.step = max(0, st.step - 1)
            elif key == ord("s"):
                errors = _layout_errors(st)
                if errors:
                    print("[ROI] Cannot save:")
                    for e in errors:
                        print(f"       - {e}")
                else:
                    _save_layout(load_camera_config(config_path), st, config_path)
                    saved = True
                    print(f"[ROI] Layout saved to {config_path}")
                    break
            elif key == 27:  # Esc
                print("[ROI] Cancelled - nothing saved.")
                break
    finally:
        cv2.destroyWindow(window)
    return saved


def apply_layout_to_config(config_path, roi, line, side, shape_w, shape_h) -> CameraConfig:
    """Programmatic layout write (used by tests / scripts)."""
    config = load_camera_config(config_path)
    config.roi = [[float(x), float(y)] for x, y in roi]
    config.counting_line = [[float(x), float(y)] for x, y in line]
    config.xray_side_point = [float(side[0]), float(side[1])]
    config.source_shape_w = int(shape_w)
    config.source_shape_h = int(shape_h)
    config.save(config_path)
    return config