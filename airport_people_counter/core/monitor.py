"""QueueMonitor — the engine orchestrator.

Ingests sensor readings and passenger events, runs the fusion → metrics →
state-machine → backlog → alert pipeline, and exposes a JSON-friendly
``ZoneSnapshot`` that the REST/WebSocket API and dashboard will consume.

Kept pure-Python so it runs identically against the live vision pipeline
(Sensor 1) or the simulation harness.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from .alerts import AlertEngine
from .backlog import BacklogDetector
from .config import Settings
from .fusion import FusionResult, SensorFusion
from .metrics import (density, occupancy_percentage, waiting_time)
from .models import (EventType, MetricsSnapshot, PassengerEvent,
                     QueueState, SensorHealth, SensorReading, ZoneSnapshot,
                     iso_to_unix, utc_now_iso)
from .state_machine import QueueStateMachine


class QueueMonitor:
    def __init__(self, settings: Optional[Settings] = None, config_path=None, clock=None):
        self.settings = settings or Settings.load(config_path)
        self._clock = clock or utc_now_iso
        self.fusion = SensorFusion(self.settings, clock=self._clock)
        self.state_machine = QueueStateMachine(self.settings)
        self.backlog = BacklogDetector(self.settings, clock=self._clock)
        self.alerts = AlertEngine(self.settings)

        # passenger event tracking
        self._events: deque = deque(maxlen=10000)
        self._in_queue: dict[str, float] = {}  # tracking_id -> entry unix

        # congestion stopwatch
        self._congestion_start: Optional[float] = None

        self._last_state = QueueState.DATA_UNAVAILABLE

    # ── ingestion ────────────────────────────────────────────
    def ingest_sensor(self, reading: SensorReading) -> None:
        self.fusion.ingest(reading)

    def ingest_passenger_event(self, event: PassengerEvent) -> None:
        if event.server_received_time is None:
            event.server_received_time = self._clock()
        self._events.append(event)
        ts = iso_to_unix(event.event_time)
        if event.event_type == EventType.ENTER:
            if event.tracking_id:
                self._in_queue[event.tracking_id] = ts
        elif event.event_type == EventType.EXIT:
            if event.tracking_id:
                self._in_queue.pop(event.tracking_id, None)

    # ── helpers ──────────────────────────────────────────────
    def _unix_now(self) -> float:
        return iso_to_unix(self._clock())

    def _avg_waiting_time(self) -> float:
        now = self._unix_now()
        waits = [now - t for t in self._in_queue.values()]
        if not waits:
            return 0.0
        return sum(waits) / len(waits)

    def _fused_people_count(self, fusion: FusionResult) -> int:
        counts = [
            r.people_count for r in fusion.sensors.values()
            if r.health == SensorHealth.ONLINE and not self._is_downstream(r.sensor_id)
        ]
        if not counts:
            for r in fusion.sensors.values():
                if r.health == SensorHealth.ONLINE:
                    counts.append(r.people_count)
                    break
        if not counts:
            return 0
        return int(round(sum(counts) / len(counts)))

    def _fused_flow(self, fusion: FusionResult) -> float:
        """Average flow-rate from online primary sensors."""
        values = [
            r.flow_rate for r in fusion.sensors.values()
            if r.health == SensorHealth.ONLINE and not self._is_downstream(r.sensor_id)
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _is_downstream(self, sensor_id: str) -> bool:
        for spec in self.settings.sensors.specs.values():
            if spec.id == sensor_id:
                return spec.role == "downstream_xray_validation"
        return False

    # ── congestion stopwatch ─────────────────────────────────
    def _update_congestion(self, state: QueueState) -> float:
        """Duration of the *current* congestion episode (0 when not congested)."""
        now = self._unix_now()
        congested = state in (QueueState.CONGESTED, QueueState.CRITICAL, QueueState.BACKLOG)
        if congested:
            if self._congestion_start is None:
                self._congestion_start = now
            return now - self._congestion_start
        self._congestion_start = None
        return 0.0

    # ── evaluation ───────────────────────────────────────────
    def evaluate(self) -> ZoneSnapshot:
        now = self._clock()
        fusion = self.fusion.evaluate()
        zone = self.settings.zone

        people = self._fused_people_count(fusion)
        density_value = density(people, zone.area_m2)
        occupancy = occupancy_percentage(people, zone.max_capacity)
        flow = self._fused_flow(fusion)
        wait = self._avg_waiting_time()

        queue_people = people
        if self._in_queue:
            queue_people = len(self._in_queue)

        metrics = MetricsSnapshot(
            timestamp=now,
            people_count=people,
            density=density_value,
            occupancy_percent=occupancy,
            average_distance=0.0,
            flow_rate=flow,
            average_waiting_time_s=wait,
            queue_people=queue_people,
            tracked_count=people,
        )

        # Override people_count when every sensor is offline/degraded
        if fusion.any_offline and not fusion.primary_ok and not fusion.sensors:
            metrics.people_count = 0
            density_value = 0.0
            metrics.density = 0.0

        state = self.state_machine.update(metrics)
        congestion_s = self._update_congestion(state)
        metrics.congestion_duration_s = congestion_s

        backlog = self.backlog.update(queue_people, density_value, flow)

        # Special states take precedence over the ladder classification.
        # Precedence: no data > disagreement > backlog > ladder state.
        if not fusion.primary_ok:
            state = (QueueState.SENSOR_ERROR if fusion.primary_error
                     else QueueState.DATA_UNAVAILABLE)
        elif any(m in ("people_count", "density") for m, *_ in fusion.disagreements):
            state = QueueState.SENSOR_DISAGREEMENT
        elif backlog.active:
            state = QueueState.BACKLOG

        confidence = fusion.agreement_confidence

        open_alerts = self.alerts.evaluate(
            state=state,
            previous_state=self._last_state,
            sensors=fusion.sensors,
            disagreement=fusion.disagreements,
            backlog_active=backlog.active,
            confidence=confidence,
            metrics=metrics,
        )
        self._last_state = state

        return ZoneSnapshot(
            zone_id=self.settings.zone_id,
            terminal_id=self.settings.terminal_id,
            airport_id=self.settings.airport_id,
            timestamp=now,
            state=state,
            metrics=metrics,
            sensor_states=fusion.sensors,
            agreement_confidence=confidence,
            disagreement=[list(d) for d in fusion.disagreements],
            downstream={
                "ok": fusion.downstream_ok,
                "unavailable": fusion.downstream_unavailable,
                "congested": fusion.downstream_congested,
            },
            backlog=backlog.active,
            backlog_reason=backlog.reason,
            open_alerts=[a.__dict__ for a in open_alerts],
        )

    def snapshot(self) -> ZoneSnapshot:
        return self.evaluate()
