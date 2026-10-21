"""Core domain data models.

All events carry both ``event_time`` (when the thing happened) and
``server_received_time`` (when the backend saw it), so clock-skew and
network latency are never conflated. Timestamps are ISO 8601 UTC with a
trailing ``Z``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string with Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iso_to_unix(iso: str) -> float:
    """Parse an ISO 8601 timestamp to Unix epoch seconds."""
    value = iso.replace("Z", "+00:00")
    return datetime.fromisoformat(value).timestamp()


class QueueState(str, Enum):
    """Operational state of the monitored queue area."""

    EMPTY = "EMPTY"
    SPARSE = "SPARSE"
    NORMAL = "NORMAL"
    CROWDED = "CROWDED"
    CONGESTED = "CONGESTED"
    BACKLOG = "BACKLOG"
    CRITICAL = "CRITICAL"
    SENSOR_ERROR = "SENSOR_ERROR"
    SENSOR_DISAGREEMENT = "SENSOR_DISAGREEMENT"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


class SensorHealth(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


class AlertStatus(str, Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"


@dataclass
class SensorReading:
    """A single telemetry sample from one sensor."""

    sensor_id: str
    sensor_type: str
    event_time: str
    server_received_time: str
    people_count: int = 0
    density: float = 0.0
    average_distance: float = 0.0
    flow_rate: float = 0.0
    confidence: float = 1.0
    health: SensorHealth = SensorHealth.ONLINE
    extra: dict = field(default_factory=dict)


@dataclass
class PassengerEvent:
    """A passenger entering or leaving the monitored zone."""

    zone_id: str
    event_time: str
    event_type: EventType
    tracking_id: Optional[str] = None
    server_received_time: Optional[str] = None
    confidence: float = 1.0
    position: Optional[dict] = field(default_factory=dict)  # {x,y,z} when available


@dataclass
class Alert:
    """An operational alert with a full lifecycle."""

    alert_id: str
    type: str
    timestamp: str
    severity: Severity
    message: str
    status: AlertStatus = AlertStatus.OPEN
    sensor_id: Optional[str] = None
    location: str = ""
    confidence: float = 1.0
    description: str = ""
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None


@dataclass
class MetricsSnapshot:
    """Derived metrics for a zone at an instant."""

    timestamp: str
    people_count: int = 0
    density: float = 0.0
    occupancy_percent: float = 0.0
    average_distance: float = 0.0
    flow_rate: float = 0.0
    average_waiting_time_s: float = 0.0
    congestion_duration_s: float = 0.0
    queue_people: int = 0
    tracked_count: int = 0


@dataclass
class ZoneSnapshot:
    """A full operational view of the zone (what the dashboard/API consume)."""

    zone_id: str
    terminal_id: str
    airport_id: str
    timestamp: str
    state: QueueState
    metrics: MetricsSnapshot
    sensor_states: dict = field(default_factory=dict)  # key -> SensorReading
    agreement_confidence: float = 0.0
    disagreement: list = field(default_factory=list)
    downstream: dict = field(default_factory=dict)  # {ok, unavailable, congested}
    backlog: bool = False
    backlog_reason: str = ""
    open_alerts: list = field(default_factory=list)
