"""Centralized, per-zone operational configuration.

Loads a YAML file (see ``config/default.yaml``) into typed dataclasses so
thresholds and tolerances are configurable rather than hard-coded. A
programmatic ``Settings`` built from plain dicts is also supported, which
keeps the engine dependency-free for tests and simulation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class ZoneConfig:
    name: str = "Queue zone"
    area_m2: float = 50.0
    max_capacity: int = 50
    warning_percentage: float = 70.0
    crowded_percentage: float = 85.0
    critical_percentage: float = 100.0


@dataclass
class DensityThresholds:
    empty: float = 0.00
    sparse: float = 0.10
    normal: float = 0.40
    crowded: float = 0.60
    congested: float = 0.80


@dataclass
class FlowConfig:
    baseline_per_minute: float = 12.0
    backlog_ratio: float = 0.5
    windows_minutes: list = field(default_factory=lambda: [1, 5, 15, 30, 60])


@dataclass
class DistanceConfig:
    safe_spacing_m: float = 1.0
    calibration_px_per_m: float = 100.0


@dataclass
class SensorSpec:
    id: str = ""
    type: str = "vision"
    role: str = "unknown"
    enabled: bool = True


@dataclass
class AgreementTolerance:
    people_count: float = 0.15
    density: float = 0.20
    flow_rate: float = 0.25


@dataclass
class SensorsConfig:
    specs: dict = field(default_factory=dict)  # key -> SensorSpec
    agreement_tolerance: AgreementTolerance = field(default_factory=AgreementTolerance)
    staleness_seconds: float = 5.0

    def enabled_ids(self) -> list:
        return [k for k, s in self.specs.items() if s.enabled]


@dataclass
class StateConfig:
    debounce_frames: int = 5
    hysteresis: float = 0.05


@dataclass
class BacklogConfig:
    min_queue_people: int = 15
    min_density: float = 0.60
    persistence_seconds: float = 10.0


@dataclass
class AlertsConfig:
    enabled: bool = True
    min_confidence: float = 0.60
    debounce_seconds: float = 5.0


@dataclass
class PerformanceTargets:
    telemetry_latency_s: float = 1.0
    dashboard_latency_s: float = 2.0
    alert_latency_s: float = 3.0


@dataclass
class Settings:
    airport_id: str = "A01"
    terminal_id: str = "T1"
    zone_id: str = "XRAY01"
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    density_thresholds: DensityThresholds = field(default_factory=DensityThresholds)
    flow: FlowConfig = field(default_factory=FlowConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    state: StateConfig = field(default_factory=StateConfig)
    backlog: BacklogConfig = field(default_factory=BacklogConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    performance_targets: PerformanceTargets = field(default_factory=PerformanceTargets)

    # ── loaders ─────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        return cls(
            airport_id=str(data.get("airport_id", "A01")),
            terminal_id=str(data.get("terminal_id", "T1")),
            zone_id=str(data.get("zone_id", "XRAY01")),
            zone=ZoneConfig(**data.get("zone", {})),
            density_thresholds=DensityThresholds(**data.get("density_thresholds", {})),
            flow=FlowConfig(**data.get("flow", {})),
            distance=DistanceConfig(**data.get("distance", {})),
            sensors=_parse_sensors(data.get("sensors", {})),
            state=StateConfig(**data.get("state", {})),
            backlog=BacklogConfig(**data.get("backlog", {})),
            alerts=AlertsConfig(**data.get("alerts", {})),
            performance_targets=PerformanceTargets(**data.get("performance_targets", {})),
        )

    @classmethod
    def from_file(cls, path) -> "Settings":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def load(cls, path=None) -> "Settings":
        """Load from a file, or the default config when no path is given."""
        if path is None:
            path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
        if Path(path).exists():
            return cls.from_file(path)
        return cls()


def _parse_sensors(data: dict) -> SensorsConfig:
    specs = {}
    for key, raw in (data or {}).items():
        if not isinstance(raw, dict):
            continue
        if key == "agreement_tolerance":
            continue
        if key == "staleness_seconds":
            continue
        specs[key] = SensorSpec(**raw)
    return SensorsConfig(
        specs=specs,
        agreement_tolerance=AgreementTolerance(**data.get("agreement_tolerance", {})),
        staleness_seconds=float(data.get("staleness_seconds", 5.0)),
    )
