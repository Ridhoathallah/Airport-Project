"""Mapping schema and geometry for camera-relative queue calibration.

This module owns the *mapping* — the ROI polygon plus the separate IN and
OUT virtual counting lines a technician draws on a camera frame — and the
conversions between frame-normalized ``[0, 1]`` coordinates and native
frame pixels.

Only ``Load``/``Save`` need the disk: geometry, rendering and crossing
mathematics are free of cv2 / OpenCV side effects so that they can be
unit-tested against real arithmetic.

Design decisions (equivalent to the approved design brief):
  * Coordinates are stored normalised to the range ``[0, 1]`` so the
    mapping survives a camera resolution change and can be rendered with
    correct aspect ratio at any viewport size.
  * A mapping is versioned (``schema_version``) and keyed by ``camera_id``
    so a future multi-camera deployment can persist one entry per camera
    without a schema change.
  * A valid new mapping is the *single active geometry source*: while it
    is configured, the legacy ``counting_line``/``xray_side_point``
    geometry in ``config/camera.yaml`` is not used. Legacy geometry is
    only applied when no mapping exists for that camera, as a fallback.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

MAPPING_SCHEMA_VERSION = 1

# Per-line direction targets. A line points *into* its positive half-side
# for the lower-numbered side and *out of* the positive half-side for the
# higher-numbered side, which is the same convention the numeric
# `_line_side` helper uses. They stay explicit here so a duplicated line
# can differ only in direction, which is what a technician expects.
IN_DIRECTION = "IN"
OUT_DIRECTION = "OUT"
VALID_DIRECTIONS = (IN_DIRECTION, OUT_DIRECTION)

# Supported reference points. ``bottom_center`` is the calibrated default
# for the overhead/high-angle cameras this feature targets: the bottom of
# the bounding box is the point that stays closest to the real footprint,
# while a centroid "floats" above the ground and drifts as soon as a body
# leans or extends beyond the frame edge. ``centroid`` is retained as a
# compatibility fallback for pipelines whose ROI was calibrated against
# the legacy centroid semantics.
BOTTOM_CENTER_REFERENCE = "bottom_center"
CENTROID_REFERENCE = "centroid"
VALID_REFERENCE_POINTS = (BOTTOM_CENTER_REFERENCE, CENTROID_REFERENCE)

# Default metadata used when a mapping has no recorded values yet.
DEFAULT_CAMERA_ID = "CAMERA 01"


class MappingError(ValueError):
    """Raised when a mapping cannot be loaded, validated or applied."""


# ---------------------------------------------------------------------
#  SCHEMA
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LineMapping:
    """One virtual counting line: exactly two endpoints plus a direction."""

    points: Tuple[Tuple[float, float], Tuple[float, float]]
    direction: str

    @staticmethod
    def from_list(data) -> List["LineMapping"]:
        """Build ``LineMapping`` objects from a raw decoded YAML list."""
        if not isinstance(data, list) or not data:
            raise MappingError("lines must be a non-empty list")
        out: List[LineMapping] = []
        for raw in data:
            if not isinstance(raw, dict):
                raise MappingError("each line entry must be a mapping")
            pts = raw.get("points")
            if not isinstance(pts, list) or len(pts) != 2:
                raise MappingError("each line must have exactly 2 endpoints")
            direction = str(raw.get("direction", "")).upper()
            if direction not in VALID_DIRECTIONS:
                raise MappingError(
                    f"line direction must be one of {VALID_DIRECTIONS}, got {direction!r}"
                )
            try:
                a, b = pts[0], pts[1]
                out.append(
                    LineMapping(
                        points=((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))),
                        direction=direction,
                    )
                )
            except (TypeError, ValueError, IndexError, KeyError) as exc:
                raise MappingError(f"malformed line points {pts!r}: {exc}") from exc
        return out


@dataclass
class MappingGeometry:
    """A validated, normalised mapping for one camera."""

    camera_id: str = DEFAULT_CAMERA_ID
    schema_version: int = MAPPING_SCHEMA_VERSION
    roi: Tuple[Tuple[float, float], ...] = ()
    lines: Dict[str, LineMapping] = field(default_factory=dict)
    reference_point: str = BOTTOM_CENTER_REFERENCE
    source_shape: Optional[Tuple[int, int]] = None  # (width, height)
    created_at_iso: Optional[str] = None
    updated_at_iso: Optional[str] = None

    # -- helpers used by both the editing UI and the analytics engine --

    @property
    def in_line(self) -> Optional[LineMapping]:
        return self.lines.get("in")

    @property
    def out_line(self) -> Optional[LineMapping]:
        return self.lines.get("out")

    @property
    def configured(self) -> bool:
        """A mapping counts as configured only when it is fully usable."""
        return bool(self.roi) and bool(self.lines)

    def to_dict(self) -> dict:
        """Serializable form, normalised coordinates only."""
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "roi": [[float(x), float(y)] for x, y in self.roi],
            "lines": [
                {
                    "id": ident,
                    "points": [[float(x), float(y)] for x, y in line.points],
                    "direction": line.direction,
                }
                for (ident, line) in sorted(self.lines.items())
            ],
            "reference_point": self.reference_point,
            "source_shape": (
                {"width": int(self.source_shape[0]), "height": int(self.source_shape[1])}
                if self.source_shape
                else None
            ),
            "created_at_iso": self.created_at_iso,
            "updated_at_iso": self.updated_at_iso,
        }

    @classmethod
    def from_dict(cls, data: dict, camera_id: str = DEFAULT_CAMERA_ID) -> "MappingGeometry":
        """Parse a mapping entry, raising ``MappingError`` on malformed input."""
        if not isinstance(data, dict):
            raise MappingError("mapping entry must be a YAML mapping")
        version = data.get("schema_version")
        if not isinstance(version, int):
            raise MappingError("schema_version must be an integer")
        if version != MAPPING_SCHEMA_VERSION:
            raise MappingError(
                f"unsupported mapping schema_version {version!r}, "
                f"expected {MAPPING_SCHEMA_VERSION}"
            )
        mapping = cls(camera_id=str(data.get("camera_id") or camera_id), schema_version=version)

        roi = data.get("roi")
        if not isinstance(roi, list) or len(roi) < 3:
            raise MappingError("roi must be a list with at least 3 vertices")
        try:
            mapping.roi = tuple((float(p[0]), float(p[1])) for p in roi)
        except (TypeError, ValueError, IndexError) as exc:
            raise MappingError(f"malformed roi vertices: {exc}") from exc

        raw_lines = data.get("lines")
        if raw_lines is None:
            mapping.lines = {}
        else:
            if isinstance(raw_lines, dict):
                # Canonical persisted form: ``in:`` / ``out:`` keys, each
                # with ``points`` and ``direction``. Normalised into the
                # list form below so both shapes parse identically.
                normalised: List[dict] = []
                for ident, body in raw_lines.items():
                    if not isinstance(body, dict):
                        raise MappingError(f"line {ident!r} must be a YAML mapping")
                    body = dict(body)
                    body["id"] = str(ident)
                    normalised.append(body)
                raw_lines = normalised
            if not isinstance(raw_lines, list) or not raw_lines:
                raise MappingError("lines must be a non-empty list or in/out mapping")
            for line in LineMapping.from_list(raw_lines):
                mapping.lines[_line_index(raw_lines, line)] = line
        if not mapping.lines:
            raise MappingError("at least one counting line must be configured")

        reference = str(data.get("reference_point", BOTTOM_CENTER_REFERENCE) or "").lower()
        if reference not in VALID_REFERENCE_POINTS:
            raise MappingError(
                f"reference_point must be one of {VALID_REFERENCE_POINTS}, got {reference!r}"
            )
        mapping.reference_point = reference

        shape = data.get("source_shape") or {}
        if isinstance(shape, dict) and shape.get("width") and shape.get("height"):
            mapping.source_shape = (int(shape["width"]), int(shape["height"]))

        mapping.created_at_iso = data.get("created_at_iso")
        mapping.updated_at_iso = data.get("updated_at_iso")
        return mapping


def _line_index(raw_lines, line: LineMapping) -> str:
    """Return the id the ``raw_lines`` entry used, or derive one."""
    for raw in raw_lines:
        pts = raw.get("points")
        try:
            match = pts and _same_point(pts[0], line.points[0]) and _same_point(pts[1], line.points[1])
        except (TypeError, IndexError):
            match = False
        if match and raw.get("direction", "").upper() == line.direction:
            ident = str(raw.get("id") or "").lower()
            if ident in ("in", "out"):
                return ident
            # Derive a canonical id from the direction when no id is given.
            return line.direction.lower()
    return line.direction.lower()


def _same_point(a, b, eps: float = 1e-9) -> bool:
    try:
        return bool(abs(float(a[0]) - float(b[0])) <= eps and abs(float(a[1]) - float(b[1])) <= eps)
    except (TypeError, IndexError, ValueError):
        return False


# ---------------------------------------------------------------------
#  VALIDATION
# ---------------------------------------------------------------------


def _point_in_unit_range(p: Tuple[float, float]) -> bool:
    x, y = p
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def _line_length_frames(line: LineMapping, shape: Optional[Tuple[int, int]]) -> Optional[float]:
    """Line length expressed in native frame pixels, ``None`` if unknown."""
    if shape is None:
        return None
    w, h = shape
    (x1, y1), (x2, y2) = line.points
    return float(((x2 * w - x1 * w) ** 2 + (y2 * h - y1 * h) ** 2) ** 0.5)


def _poly_area(poly) -> float:
    """Shoelace area; ``>>> 0`` only for a non-degenerate polygon."""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Orientation-based segment intersection (with degenerate handling)."""

    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else -1

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, p3):
        return True
    if o2 == 0 and on_segment(p1, p2, p4):
        return True
    if o3 == 0 and on_segment(p3, p4, p1):
        return True
    if o4 == 0 and on_segment(p3, p4, p2):
        return True
    return False


def is_simple_polygon(poly) -> bool:
    """True when a closed polyline has no crossing edges."""
    n = len(poly)
    if n < 3:
        return False
    edges = [(poly[i], poly[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        a1, b1 = edges[i]
        for j in range(i + 1, n):
            a2, b2 = edges[j]
            # Adjacent edges legitimately share a vertex — that is not a
            # self-intersection. Only report crossings between edges that
            # do not touch.
            if (j - i) in (1, n - 1):
                continue
            if _segments_intersect(a1, b1, a2, b2):
                return False
    return True


#: A counting line shorter than this fraction of the frame diagonal
#: cannot reliably separate two sides of a queue — points jitter more
#: than that between frames at monitoring feeds.
MIN_LINE_FRACTION_OF_DIAGONAL = 0.01


def validate_mapping(mapping: MappingGeometry, shape: Optional[Tuple[int, int]] = None) -> List[str]:
    """Return a list of human-readable validation errors (empty = valid).

    ``shape`` is ``(width, height)`` when known; it enables the
    deterministic physical checks (line pixel length) that cannot be
    performed on normalised coordinates alone.
    """
    errors: List[str] = []
    if not isinstance(mapping, MappingGeometry):
        errors.append("mapping must be a MappingGeometry")
        return errors

    if mapping.schema_version != MAPPING_SCHEMA_VERSION:
        errors.append(
            f"schema_version {mapping.schema_version} is unsupported "
            f"(expected {MAPPING_SCHEMA_VERSION})"
        )
    if mapping.reference_point not in VALID_REFERENCE_POINTS:
        errors.append(f"reference_point {mapping.reference_point!r} is not supported")

    # ---- ROI -------------------------------------------------------
    if not mapping.roi or len(mapping.roi) < 3:
        errors.append("ROI polygon must have at least 3 vertices")
        return errors
    for i, p in enumerate(mapping.roi):
        if not _point_in_unit_range(p):
            errors.append(f"ROI vertex #{i} {p} is outside the [0, 1] frame range")
    if _poly_area(mapping.roi) <= 0:
        errors.append("ROI polygon is degenerate (zero area)")
    if not is_simple_polygon(mapping.roi):
        errors.append("ROI polygon self-intersects")

    # ---- counting lines -------------------------------------------
    if not mapping.lines:
        errors.append("at least one counting line must be configured")
        return errors
    diagonal = None
    if shape is not None and shape[0] > 0 and shape[1] > 0:
        diagonal = (float(shape[0]) ** 2 + float(shape[1]) ** 2) ** 0.5
    min_len_px = MIN_LINE_FRACTION_OF_DIAGONAL * diagonal if diagonal else None
    for ident, line in sorted(mapping.lines.items()):
        if len(line.points) != 2:
            errors.append(f"line {ident!r} must have exactly 2 endpoints")
            continue
        for i, p in enumerate(line.points):
            if not _point_in_unit_range(p):
                errors.append(f"line {ident!r} endpoint #{i} {p} is outside the [0, 1] frame range")
        if _same_point(line.points[0], line.points[1]):
            errors.append(f"line {ident!r} endpoints are identical (degenerate line)")
        length_px = _line_length_frames(line, shape)
        if min_len_px is not None and length_px is not None and length_px < min_len_px:
            errors.append(
                f"line {ident!r} is too short "
                f"({length_px:.1f} px < {min_len_px:.1f} px in the {shape[0]}x{shape[1]} frame)"
            )
        if line.direction not in VALID_DIRECTIONS:
            errors.append(f"line {ident!r} direction {line.direction!r} is not supported")

    # ---- semantic comparison between the IN and OUT lines ----------
    if "in" in mapping.lines and "out" in mapping.lines:
        in_line = mapping.lines["in"]
        out_line = mapping.lines["out"]
        same_pts = _same_point(in_line.points[0], out_line.points[0]) and _same_point(
            in_line.points[1], out_line.points[1]
        )
        if same_pts and in_line.direction == out_line.direction:
            errors.append(
                f"IN and OUT lines are identical (same endpoints and direction {in_line.direction!r})"
            )

    return errors


# ---------------------------------------------------------------------
#  COORDINATE CONVERSION
# ---------------------------------------------------------------------


@dataclass
class PixelGeometry:
    """Mapping geometry expressed in native frame pixels for one frame."""

    roi: Tuple[Tuple[float, float], ...] = ()
    in_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    out_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    in_direction: Optional[str] = None
    out_direction: Optional[str] = None
    reference_point: str = BOTTOM_CENTER_REFERENCE


def to_pixel(mapping: MappingGeometry, shape: Tuple[int, int]) -> PixelGeometry:
    """Convert a normalised mapping to native pixel coordinates.

    ``shape`` is ``(width, height)``, matching the rest of the codebase
    (e.g. the ``QueueAnalytics.update(..., shape)`` argument).
    """
    w, h = shape
    if w <= 0 or h <= 0:
        raise MappingError(f"invalid frame shape {shape!r}")
    roi = tuple((round(float(x) * w, 4), round(float(y) * h, 4)) for x, y in mapping.roi)
    out = PixelGeometry(roi=roi, reference_point=mapping.reference_point)
    for ident in ("in", "out"):
        line = mapping.lines.get(ident)
        if line is None:
            continue
        a = (round(float(line.points[0][0]) * w, 4), round(float(line.points[0][1]) * h, 4))
        b = (round(float(line.points[1][0]) * w, 4), round(float(line.points[1][1]) * h, 4))
        setattr(out, f"{ident}_line", (a, b))
        setattr(out, f"{ident}_direction", line.direction)
    return out


def from_pixel(
    roi,
    lines: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float], str]]],
    shape: Tuple[int, int],
    camera_id: str = DEFAULT_CAMERA_ID,
    reference_point: str = BOTTOM_CENTER_REFERENCE,
) -> MappingGeometry:
    """Convert native pixel geometry into a normalised mapping.

    Used by the technician-facing pipeline and the one-shot legacy
    migration helper. ``lines`` is a dictionary, e.g.::

        lines={"in": ((x1, y1), (x2, y2), "IN"), "out": ((...) , (…), "OUT")}
    """
    w, h = shape
    if w <= 0 or h <= 0:
        raise MappingError(f"invalid frame shape {shape!r}")
    mapping = MappingGeometry(camera_id=camera_id, reference_point=reference_point, source_shape=(w, h))
    mapping.roi = tuple((round(float(x) / w, 9), round(float(y) / h, 9)) for x, y in roi)
    for ident, (a, b, direction) in (lines or {}).items():
        mapping.lines[ident] = LineMapping(
            points=((round(float(a[0]) / w, 9), round(float(a[1]) / h, 9)),
                    (round(float(b[0]) / w, 9), round(float(b[1]) / h, 9))),
            direction=direction,
        )
    return mapping


def reference_xy(
    box: Tuple[float, float, float, float], reference_point: str
) -> Tuple[float, float]:
    """Return the tracking reference point of a ``(x1, y1, x2, y2)`` box."""
    x1, y1, x2, y2 = box
    if reference_point == CENTROID_REFERENCE:
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # ``BOTTOM_CENTER_REFERENCE`` is the default.
    return (x1 + x2) / 2.0, float(y2)


# ---------------------------------------------------------------------
#  LEGACY COMPATIBILITY (camera.yaml counting_line / xray_side_point)
# ---------------------------------------------------------------------


def from_legacy_camera_config(
    roi,
    counting_line,
    xray_side_point,
    shape: Tuple[int, int],
    camera_id: str = DEFAULT_CAMERA_ID,
    reference_point: str = CENTROID_REFERENCE,
) -> MappingGeometry:
    """Derive a ``MappingGeometry`` from the legacy camera.yaml geometry.

    Legacy semantics: crossing *toward* ``xray_side_point`` is the EXIT
    direction and crossing *away* from it is the ENTER direction. The
    derived mapping therefore carries one line with the EXIT direction
    and one mirrored line with the ENTER direction. Behaviour identical
    to the pre-mapping implementation is a hard requirement here.

    ``reference_point`` defaults to ``centroid`` because that is what the
    legacy code path used; a migrated mapping keeps behaving exactly as
    before unless the operator explicitly calibrated against
    ``bottom_center``.
    """
    if shape is None or len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise MappingError(f"invalid legacy frame shape {shape!r}")
    w, h = shape
    mapping = MappingGeometry(
        camera_id=camera_id,
        reference_point=reference_point,
        source_shape=(w, h),
    )
    mapping.roi = tuple((round(float(x) / w, 9), round(float(y) / h, 9)) for x, y in roi)
    a = (round(float(counting_line[0][0]) / w, 9), round(float(counting_line[0][1]) / h, 9))
    b = (round(float(counting_line[1][0]) / w, 9), round(float(counting_line[1][1]) / h, 9))
    # Legacy semantics: the *positive* half-plane of `_line_side` for the
    # OUT line must be the side that shares its sign with
    # `xray_side_point` (crossing towards the X-ray side == EXIT). Which
    # endpoint ordering produces that is geometry-dependent (a horizontal
    # and a vertical counting line have opposite `(a, b)`-to-side
    # correspondences), so derive it from the sign at the X-ray point:
    xb, yb = xray_side_point
    line = (a, b) if _line_side_px(a, b, float(xb) / w, float(yb) / h) >= 0 else (b, a)
    mapping.lines["out"] = LineMapping(points=line, direction=OUT_DIRECTION)
    mapping.lines["in"] = LineMapping(points=(line[1], line[0]), direction=IN_DIRECTION)
    return mapping


def _line_side_px(a, b, px: float, py: float) -> float:
    """Signed cross product in normalised coordinates (same sign rule)."""
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


# ---------------------------------------------------------------------
#  PERSISTENCE (atomic write, backup, rollback)
# ---------------------------------------------------------------------


def default_mappings_path(camera_config_path: Optional[Path] = None) -> Path:
    """Path of the mappings.yaml that belongs to ``camera.yaml``."""
    base = camera_config_path or Path(__file__).resolve().parent.parent / "config"
    base = Path(base)
    if base.suffix:
        base = base.parent
    return base / "mappings.yaml"


def load_mappings_document(path: Path) -> dict:
    """Load the whole mappings.yaml document as a plain dictionary."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise MappingError(f"{path} does not contain a YAML mapping")
    return data


def load_mapping(path: Path, camera_id: str = DEFAULT_CAMERA_ID) -> Optional[MappingGeometry]:
    """Load the mapping persisted for ``camera_id`` (or ``None``)."""
    doc = load_mappings_document(path)
    cameras = doc.get("cameras") or {}
    if not isinstance(cameras, dict):
        raise MappingError(f"{path}: 'cameras' must be a YAML mapping keyed by camera_id")
    entry = cameras.get(camera_id)
    if entry is None:
        # A deployment may key cameras by a bare string; tolerate it.
        if camera_id in doc:
            entry = doc.get(camera_id)
    if entry is None:
        return None
    # ``schema_version`` is documented at document level but must also be
    # present per camera entry so each camera can migrate independently;
    # fall back to the document version for hand-written files.
    if isinstance(entry, dict) and "schema_version" not in entry:
        entry = dict(entry)
        entry["schema_version"] = doc.get("schema_version", MAPPING_SCHEMA_VERSION)
    try:
        return MappingGeometry.from_dict(entry, camera_id=camera_id)
    except MappingError as exc:
        raise MappingError(f"{path}: camera {camera_id!r}: {exc}") from exc


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_mapping(
    path: Path,
    mapping: MappingGeometry,
    *,
    backup: bool = True,
) -> Optional[Path]:
    """Persist one camera mapping, atomically, keeping a backup file.

    Returns the backup path (``…mappings.yaml.bak``) when an existing
    store was replaced, or ``None`` when the store was created fresh.

    Raises ``MappingError`` when the resulting file would not load back
    (so a partial write is never left behind, and the caller gets a
    rollback signal instead of a half-updated store).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_mappings_document(path)
    if not existing:
        existing = {"schema_version": MAPPING_SCHEMA_VERSION, "cameras": {}}
    existing.setdefault("cameras", {})
    entry = dict(mapping.to_dict())
    if not entry.get("created_at_iso"):
        entry["created_at_iso"] =_iso_now()
    entry["updated_at_iso"] = _iso_now()
    existing["cameras"][mapping.camera_id] = entry

    # Serialise, then re-parse: fail *before* touching the real file.
    rendered = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True)
    reparsed = yaml.safe_load(rendered)
    MappingGeometry.from_dict(reparsed["cameras"][mapping.camera_id], camera_id=mapping.camera_id)

    backup_path: Optional[Path] = None
    if backup and path.exists():
        backup_path = Path(str(path) + ".bak")
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        os.replace(tmp_path, path)  # atomic on POSIX and Windows
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    # Return the backup path only; ``None`` means the store was created
    # fresh, so there was nothing to roll back to.
    return backup_path
