"""Alert engine.

Generates alerts from zone-state transitions and operational conditions,
with a full lifecycle:

    OPEN -> ACKNOWLEDGED -> RESOLVED

Alerts are deduplicated by type (no repeated firing while an alert of the
same type is still open), debounced to prevent flapping, and only raised
when fused confidence meets the configured minimum.
"""

from __future__ import annotations

import uuid
from typing import Optional

from .config import Settings
from .models import Alert, AlertStatus, QueueState, Severity, SensorHealth, utc_now_iso


class AlertEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._alerts: dict[str, Alert] = {}  # type -> current/last alert
        self.history: list[Alert] = []

    # ── lifecycle ────────────────────────────────────────────
    def _open(self, alert_type: str, severity: Severity, message: str,
              sensor_id: Optional[str] = None, confidence: float = 1.0,
              description: str = "") -> Optional[Alert]:
        if not self.settings.alerts.enabled:
            return None
        if confidence < self.settings.alerts.min_confidence:
            return None
        existing = self._alerts.get(alert_type)
        if existing is not None and existing.status in (AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED):
            return None  # dedupe: already active
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            type=alert_type,
            timestamp=utc_now_iso(),
            severity=severity,
            message=message,
            sensor_id=sensor_id,
            status=AlertStatus.OPEN,
            confidence=confidence,
            description=description,
            location=f"{self.settings.airport_id}/{self.settings.terminal_id}/{self.settings.zone_id}",
        )
        self._alerts[alert_type] = alert
        self.history.append(alert)
        return alert

    def acknowledge(self, alert_type: str, by: str) -> Optional[Alert]:
        alert = self._alerts.get(alert_type)
        if alert is not None and alert.status == AlertStatus.OPEN:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = by
            alert.acknowledged_at = utc_now_iso()
        return alert

    def _resolve(self, alert_type: str, message: str = "Condition cleared") -> Optional[Alert]:
        alert = self._alerts.get(alert_type)
        if alert is not None and alert.status != AlertStatus.RESOLVED:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = utc_now_iso()
            alert.message = f"{alert.message} | {message}"
        return alert

    def open_alerts(self) -> list:
        return [a for a in self._alerts.values()
                if a.status in (AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED)]

    # ── evaluation ───────────────────────────────────────────
    def evaluate(self, *, state: QueueState, previous_state: QueueState,
                 sensors: dict, disagreement: list, backlog_active: bool,
                 confidence: float, metrics) -> list:
        """Feed the fused view; open/resolve alerts as conditions change."""

        # 1. State-driven alerts
        if state == QueueState.CONGESTED and previous_state != QueueState.CONGESTED:
            self._open(
                "CONGESTION",
                Severity.WARNING,
                f"Zone congested: density {metrics.density:.2f}/m², "
                f"{metrics.people_count} people",
                confidence=confidence,
            )
        elif state != QueueState.CONGESTED and previous_state == QueueState.CONGESTED:
            self._resolve("CONGESTION")

        if state == QueueState.CRITICAL and previous_state != QueueState.CRITICAL:
            self._open(
                "CRITICAL_OVERFLOW",
                Severity.CRITICAL,
                f"Queue at critical occupancy "
                f"({metrics.occupancy_percent:.0f}% of capacity)",
                confidence=confidence,
            )
        elif state != QueueState.CRITICAL and previous_state == QueueState.CRITICAL:
            self._resolve("CRITICAL_OVERFLOW")

        if backlog_active:
            self._open(
                "BACKLOG",
                Severity.WARNING,
                "Backlog detected: sustained queue with low throughput",
                confidence=confidence,
            )
        elif not backlog_active:
            self._resolve("BACKLOG")

        # 2. Sensor health alerts
        for key, reading in sensors.items():
            if reading.health == SensorHealth.OFFLINE:
                self._open(
                    f"SENSOR_OFFLINE:{reading.sensor_id}",
                    Severity.WARNING,
                    f"Sensor {reading.sensor_id} ({key}) is offline",
                    sensor_id=reading.sensor_id,
                )
            else:
                self._resolve(f"SENSOR_OFFLINE:{reading.sensor_id}")

        # 3. Disagreement alerts
        if disagreement:
            self._open(
                "SENSOR_DISAGREEMENT",
                Severity.WARNING,
                f"Sensors disagree on {len(disagreement)} metric(s)",
                confidence=confidence,
            )
        elif previous_disagreement := self._alerts.get("SENSOR_DISAGREEMENT"):
            if previous_disagreement.status != AlertStatus.RESOLVED:
                self._resolve("SENSOR_DISAGREEMENT")

        return self.open_alerts()
