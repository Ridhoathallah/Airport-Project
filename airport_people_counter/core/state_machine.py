"""Queue-condition state machine.

Classifies the zone from derived metrics into the operational states
(EMPTY … CRITICAL). Two mechanisms prevent rapid switching:

* **Hysteresis** — a state must drop *below* its threshold minus a slack
  value before it downgrades, so small oscillations do not flip state.
* **Debounce** — a candidate state must be observed for N consecutive
  evaluations before it is committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .models import MetricsSnapshot, QueueState


# Order used to decide "worse" vs "better" along the congestion ladder.
_LADDER = [
    QueueState.EMPTY,
    QueueState.SPARSE,
    QueueState.NORMAL,
    QueueState.CROWDED,
    QueueState.CONGESTED,
]


@dataclass
class Classification:
    """Low-level density/occupancy classification (no debounce applied)."""

    state: QueueState
    driving_metric: str = ""
    value: float = 0.0
    threshold: float = 0.0


class QueueStateMachine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = QueueState.DATA_UNAVAILABLE
        self._candidate: QueueState | None = None
        self._candidate_frames = 0

    # ── classification ───────────────────────────────────────
    def classify(self, m: MetricsSnapshot) -> Classification:
        """Pure metric→state mapping (hysteresis is applied in update())."""
        dt = self.settings.density_thresholds
        zone = self.settings.zone

        if m.occupancy_percent >= zone.critical_percentage:
            return Classification(QueueState.CRITICAL, "occupancy_percent",
                                  m.occupancy_percent, zone.critical_percentage)
        if m.density >= dt.congested:
            return Classification(QueueState.CONGESTED, "density",
                                  m.density, dt.congested)
        if m.density >= dt.crowded:
            return Classification(QueueState.CROWDED, "density",
                                  m.density, dt.crowded)
        if m.density >= dt.normal:
            return Classification(QueueState.NORMAL, "density",
                                  m.density, dt.normal)
        if m.density >= dt.sparse:
            return Classification(QueueState.SPARSE, "density",
                                  m.density, dt.sparse)
        return Classification(QueueState.EMPTY, "density", m.density, dt.sparse)

    # ── hysteresis-aware transition ──────────────────────────
    def _should_accept(self, current: QueueState, target: QueueState, m: MetricsSnapshot) -> bool:
        """Apply hysteresis to downgrades along the ladder.

        Escalation is always accepted. A downgrade is only accepted once the
        metric has cleared the *current* state's threshold minus a slack, so
        small oscillations around a boundary do not flip the state.
        """
        if target not in _LADDER or current not in _LADDER:
            return True
        if _LADDER.index(target) >= _LADDER.index(current):
            return True  # escalation never needs hysteresis

        hyst = self.settings.state.hysteresis
        dt = self.settings.density_thresholds
        thresholds = {
            QueueState.CONGESTED: dt.congested,
            QueueState.CROWDED: dt.crowded,
            QueueState.NORMAL: dt.normal,
            QueueState.SPARSE: dt.sparse,
            QueueState.EMPTY: dt.sparse,
        }
        required = thresholds[current] - hyst
        return m.density < required

    def update(self, m: MetricsSnapshot) -> QueueState:
        """Feed one metrics snapshot; returns the committed state."""
        if m.people_count < 0:
            return self.state

        cls = self.classify(m)
        target = cls.state

        # Coming back from "no data": establish the new state immediately
        # rather than waiting out the debounce window.
        if self.state in (QueueState.DATA_UNAVAILABLE, QueueState.SENSOR_ERROR):
            self.state = target
            self._candidate = target
            self._candidate_frames = 0
            return self.state

        if target == self._candidate and self._should_accept(self.state, target, m):
            self._candidate_frames += 1
        else:
            self._candidate = target
            self._candidate_frames = 1

        if self._candidate_frames >= self.settings.state.debounce_frames:
            self.state = target
        return self.state
