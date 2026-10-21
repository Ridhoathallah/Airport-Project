"""Pure metric calculations.

These functions are deliberately side-effect free so they are trivial to
unit test. Physical distance is only produced from depth/LiDAR positions;
for camera-only feeds a calibrated pixels-per-metre conversion is used.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .models import iso_to_unix


def density(people_count: int, area_m2: float) -> float:
    """Passenger density in people per square metre."""
    if area_m2 <= 0:
        return 0.0
    return people_count / area_m2


def occupancy_percentage(current: int, capacity: int) -> float:
    """Queue occupancy as a percentage of configured capacity."""
    if capacity <= 0:
        return 0.0
    return current / capacity * 100.0


def flow_rate(processed_count: int, window_seconds: float) -> float:
    """Passenger throughput in people per minute."""
    if window_seconds <= 0:
        return 0.0
    return processed_count / (window_seconds / 60.0)


def waiting_time(entry_iso: str, now_iso: str) -> float:
    """Seconds a passenger has waited since entering the queue."""
    return max(0.0, iso_to_unix(now_iso) - iso_to_unix(entry_iso))


def processing_time(entry_iso: str, exit_iso: str) -> float:
    """Seconds a passenger spent in the monitored process."""
    return max(0.0, iso_to_unix(exit_iso) - iso_to_unix(entry_iso))


def euclidean_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Euclidean distance between two n-dimensional points."""
    return float(math.dist(p, q))


def pairwise_distances(positions: Iterable[Sequence[float]]) -> list:
    """All pairwise distances (in metres) between passenger positions.

    Positions must already be in physical units; calibration/coordinate
    transformation happens upstream in the sensor layer.
    """
    pts = [tuple(p) for p in positions]
    out: list = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            out.append(euclidean_distance(pts[i], pts[j]))
    return out


def distance_stats(distances: Sequence[float]):
    """Summary stats over a set of pairwise distances."""
    if not distances:
        return {
            "count": 0,
            "minimum": 0.0,
            "maximum": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "percent_below_safe_spacing": 0.0,
        }
    arr = np.asarray(distances, dtype=float)
    return {
        "count": int(arr.size),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "percent_below_safe_spacing": 0.0,  # filled by caller with the safe spacing
    }


def percent_below_spacing(distances: Sequence[float], safe_spacing_m: float) -> float:
    """Fraction (0–100) of pairwise distances below the safe spacing."""
    if not distances:
        return 0.0
    below = sum(1 for d in distances if d < safe_spacing_m)
    return below / len(distances) * 100.0


def pixels_to_metres(pixels: float, calibration_px_per_m: float) -> float:
    """Convert a pixel measurement to metres using a calibrated ratio."""
    if calibration_px_per_m <= 0:
        return 0.0
    return pixels / calibration_px_per_m
