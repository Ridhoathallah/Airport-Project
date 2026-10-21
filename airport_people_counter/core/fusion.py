"""Multi-sensor fusion layer.

Combines readings from Sensor 1 (queue entry/flow), Sensor 2 (validation)
and Sensor 3 (downstream X-ray area) into a single operational view.

A metric is in *agreement* when two sensors' values fall within a
configurable relative tolerance. When primary sensors disagree the zone
reports ``SENSOR_DISAGREEMENT`` instead of jumping to a congestion
verdict; when a sensor is stale it is marked ``OFFLINE`` and the zone
reports ``DATA_UNAVAILABLE`` rather than a false ``EMPTY``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .models import SensorHealth, SensorReading, utc_now_iso


@dataclass
class FusionResult:
    """Outcome of fusing the latest readings from every sensor."""

    timestamp: str
    sensors: dict = field(default_factory=dict)  # key -> SensorReading
    agreement_confidence: float = 0.0
    disagreements: list = field(default_factory=list)  # (metric, key_a, key_b, va, vb)
    downstream_ok: bool = True
    downstream_unavailable: bool = False
    downstream_congested: bool = False
    primary_ok: bool = True
    primary_error: bool = False

    @property
    def any_offline(self) -> bool:
        return any(r.health != SensorHealth.ONLINE for r in self.sensors.values())


def _relative_difference(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


class SensorFusion:
    def __init__(self, settings: Settings, clock=None):
        self.settings = settings
        self.readings: dict[str, SensorReading] = {}
        self._clock = clock or utc_now_iso

    # ── ingestion ────────────────────────────────────────────
    def ingest(self, reading: SensorReading) -> None:
        self.readings[reading.sensor_id] = reading

    def _now_unix(self, reading: SensorReading) -> float:
        from .models import iso_to_unix
        return iso_to_unix(reading.server_received_time or utc_now_iso())

    # ── evaluation ───────────────────────────────────────────
    def _refresh_health(self) -> None:
        from .models import iso_to_unix
        now = iso_to_unix(self._clock())
        stale_after = self.settings.sensors.staleness_seconds
        for rid, reading in self.readings.items():
            if now - iso_to_unix(reading.server_received_time) > stale_after:
                reading.health = SensorHealth.OFFLINE

    def _paired_values(self, metric: str, keys):
        out = {}
        for key in keys:
            r = self.readings.get(key)
            if r is not None and r.health == SensorHealth.ONLINE:
                out[key] = getattr(r, metric)
        return out

    def evaluate(self) -> FusionResult:
        self._refresh_health()
        result = FusionResult(timestamp=utc_now_iso())
        result.sensors = dict(self.readings)

        # Canonical: self.readings is keyed by sensor_id (the hardware id).
        # Map each id to its configured role; unknown external sensors are
        # treated as primary.
        id_to_role = {s.id: s.role for s in self.settings.sensors.specs.values()}
        primary = []
        downstream = []
        for rid in self.readings:
            role = id_to_role.get(rid, "primary")
            if role == "downstream_xray_validation":
                downstream.append(rid)
            else:
                primary.append(rid)

        tol = self.settings.sensors.agreement_tolerance
        tol_map = {
            "people_count": tol.people_count,
            "density": tol.density,
            "flow_rate": tol.flow_rate,
        }

        # Agreement is judged among the PRIMARY sensors only. Sensor 3
        # watches a different physical area (the X-ray machines), so its
        # counts are not comparable to the queue's; it only contributes a
        # downstream confirmation signal below.
        matched = 0
        total = 0
        for metric, t in tol_map.items():
            vals = self._paired_values(metric, primary)
            metric_keys = [k for k in vals]
            for i, a in enumerate(metric_keys):
                for b in metric_keys[i + 1:]:
                    total += 1
                    if _relative_difference(vals[a], vals[b]) <= t:
                        matched += 1
                    else:
                        result.disagreements.append((metric, a, b, vals[a], vals[b]))

        result.agreement_confidence = (matched / total) if total else 1.0

        # Downstream status
        if downstream:
            online = [k for k in downstream if self.readings[k].health == SensorHealth.ONLINE]
            if online:
                result.downstream_ok = True
                crowded_density = self.settings.density_thresholds.crowded
                result.downstream_congested = any(
                    self.readings[k].density >= crowded_density for k in online)
            else:
                result.downstream_ok = False
                result.downstream_unavailable = True

        if primary:
            online = [k for k in primary if self.readings[k].health == SensorHealth.ONLINE]
            result.primary_ok = len(online) > 0
            result.primary_error = len(online) == 0
        else:
            result.primary_ok = False

        return result
