"""Image-orientation math for camera rotation (0 / 90 / 180 / 270 CW).

Rotation is applied at the capture edge — every frame is rotated before
detection, drawing or display — so the rotated frame *is* the frame all
downstream geometry lives in. All mapping geometry is normalised to
[0, 1] and pixels appear only at conversion edges (see AGENTS.md), which
makes a 90°-step rotation an exact, aspect-independent transform:

    90° CW : (x, y) -> (1 - y, x)
    180°   : (x, y) -> (1 - x, 1 - y)
    270° CW: (x, y) -> (y, 1 - x)

When the rotation setting changes by a delta, existing geometry (ROI
polygon, counting lines, x-ray side point) is transformed by the SAME
delta, so the calibrated region keeps covering the same physical area of
the scene. Transforming a topology-preserving rigid step keeps line
directions and the in/out relationships intact.

Pure module: numpy/cv2 only for frame rotation, no pipeline imports.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

VALID_ROTATIONS = (0, 90, 180, 270)
Rotation = int  # one of VALID_ROTATIONS
NormPoint = Tuple[float, float]
PxPoint = Tuple[float, float]


def is_valid_rotation(value) -> bool:
    try:
        return int(value) == value and int(value) in VALID_ROTATIONS
    except (TypeError, ValueError):
        return False


def normalize_rotation(value) -> Rotation:
    """Coerce to a valid rotation; raises ValueError on anything else."""
    try:
        iv = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rotation must be an integer, got {value!r}") from exc
    if iv not in VALID_ROTATIONS:
        raise ValueError(f"rotation must be one of {VALID_ROTATIONS}, got {iv}")
    return iv


def delta_rotation(current: Rotation, new: Rotation) -> Rotation:
    """Clockwise step count needed to go from *current* to *new*.

    ``rotate_norm_point(p, delta_rotation(c, n))`` maps a point stored in
    the ``c``-rotated frame onto the ``n``-rotated frame covering the
    same physical scene point.
    """
    return (normalize_rotation(new) - normalize_rotation(current)) % 360


def rotate_frame(frame: np.ndarray, degrees: Rotation) -> np.ndarray:
    """Rotate a BGR frame by a clockwise 90°-multiple. 0° returns as-is."""
    d = normalize_rotation(degrees)
    if d == 0:
        return frame
    if d == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_np_rot90(frame: np.ndarray, degrees: Rotation) -> np.ndarray:
    """cv2-free variant (handy in tests/edge contexts); returns a copy."""
    d = normalize_rotation(degrees)
    if d == 0:
        return frame
    # cv2 90 CW == np.rot90 k=-1 (for HxWxC arrays)
    return np.rot90(frame, k=-d // 90).copy()


def rotate_norm_point(point: Sequence[float], degrees: Rotation) -> NormPoint:
    """Transform a normalised [0, 1] point for a clockwise rotation."""
    x, y = float(point[0]), float(point[1])
    d = normalize_rotation(degrees)
    if d == 90:
        return (1.0 - y, x)
    if d == 180:
        return (1.0 - x, 1.0 - y)
    if d == 270:
        return (y, 1.0 - x)
    return (x, y)


def rotated_size(w: int, h: int, degrees: Rotation) -> Tuple[int, int]:
    """Frame size after rotation; 90/270 swap width and height."""
    d = normalize_rotation(degrees)
    if d in (90, 270):
        return (int(h), int(w))
    return (int(w), int(h))


def rotate_px_point(
    x: float,
    y: float,
    degrees: Rotation,
    size: Tuple[float, float],
) -> PxPoint:
    """Transform a pixel-coords point recorded on a (w, h) frame.

    Goes through the normalised plane (exact for the 90°-step grid) and
    back, so the point keeps covering the same physical location on the
    rotated frame.
    """
    w, h = float(size[0]), float(size[1])
    if w <= 0 or h <= 0:
        return (float(x), float(y))
    u, v = rotate_norm_point((x / w, y / h), degrees)
    new_w, new_h = rotated_size(int(w), int(h), degrees)
    return (u * float(new_w), v * float(new_h))


# ---------------------------------------------------------------------
#  Mapping geometry transforms
# ---------------------------------------------------------------------


def rotate_mapping(mapping, degrees: Rotation):
    """Return a rotated copy of a MappingGeometry (or None)."""
    rotated_roi = tuple(rotate_norm_point(p, degrees) for p in mapping.roi)
    rotated_lines: Dict[str, object] = {}
    for name, line in mapping.lines.items():
        pts = line.points
        a = rotate_norm_point(pts[0], degrees)
        b = rotate_norm_point(pts[1], degrees)
        # direction label rides along unchanged: a rigid rotation
        # preserves the endpoints' orientation relationship.
        rotated_lines[name] = type(line)(points=(a, b), direction=line.direction)
    rotated_shape: Optional[Tuple[int, int]] = None
    if mapping.source_shape:
        rotated_shape = rotated_size(mapping.source_shape[0], mapping.source_shape[1], degrees)
    return type(mapping)(
        camera_id=mapping.camera_id,
        schema_version=mapping.schema_version,
        roi=rotated_roi,
        lines=rotated_lines,
        reference_point=mapping.reference_point,
        source_shape=rotated_shape,
    )


def transform_legacy_layout(
    roi: Optional[List],
    counting_line: Optional[List],
    xray_side_point: Optional[List],
    degrees: Rotation,
    size: Tuple[float, float],
) -> dict:
    """Transform legacy camera.yaml pixel geometry for a rotation delta."""
    def pts(seq):
        return [[float(p[0]), float(p[1])] for p in seq]

    out: dict = {}
    if roi:
        out["roi"] = [list(rotate_px_point(p[0], p[1], degrees, size)) for p in roi]
    if counting_line and len(counting_line) == 2:
        out["counting_line"] = [
            list(rotate_px_point(p[0], p[1], degrees, size)) for p in counting_line
        ]
    if xray_side_point:
        new_pt = rotate_px_point(xray_side_point[0], xray_side_point[1], degrees, size)
        out["xray_side_point"] = [new_pt[0], new_pt[1]]
    return out
