"""Backlog detection.

A backlog is a *sustained* condition, so it is detected from a
combination of signals rather than a single metric:

* queue size at or above ``min_queue_people``,
* density at or above ``min_density``,
* flow rate well below the configured baseline (``backlog_ratio``),
* the condition persisting for ``persistence_seconds``.

A separate verdict object carries the human-readable reason so the
dashboard can explain *why* a backlog was raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .models import utc_now_iso


@dataclass
class BacklogVerdict:
    active: bool = False
    reason: str = ""
    since: Optional[str] = None


class BacklogDetector:
    def __init__(self, settings: Settings, clock=None):
        self.settings = settings
        self._clock = clock or utc_now_iso
        self._active = False
        self._active_since: Optional[str] = None
        self._satisfied_since: Optional[float] = None  # unix time

    def _unix(self) -> float:
        from .models import iso_to_unix
        return iso_to_unix(self._clock())

    def update(self, queue_people: int, density_value: float, flow_per_min: float) -> BacklogVerdict:
        cfg = self.settings.backlog
        now = self._unix()

        conditions = []
        if queue_people < cfg.min_queue_people:
            conditions.append(f"queue={queue_people}<{cfg.min_queue_people}")
        if density_value < cfg.min_density:
            conditions.append(f"density={density_value:.2f}<{cfg.min_density}")
        baseline = self.settings.flow.baseline_per_minute
        low_flow = baseline * self.settings.flow.backlog_ratio
        if flow_per_min >= low_flow:
            conditions.append(f"flow={flow_per_min:.1f}>={low_flow:.1f}/min")

        # conditions lists which requirements are NOT met.
        if conditions:
            self._satisfied_since = None
            if self._active:
                self._active = False
                self._active_since = None
            return BacklogVerdict(False, reason="; ".join(conditions))

        if self._satisfied_since is None:
            self._satisfied_since = now
        persisted = (now - self._satisfied_since) >= cfg.persistence_seconds

        if persisted and not self._active:
            self._active = True
            self._active_since = self._clock()

        reason = "all backlog conditions met" if self._active else "waiting for persistence"
        return BacklogVerdict(
            active=self._active,
            reason=reason,
            since=self._active_since,
        )
