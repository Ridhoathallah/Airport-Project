"""Unit tests for backlog detection and the alert engine."""

import unittest

from core.alerts import AlertEngine, AlertStatus
from core.backlog import BacklogDetector
from core.config import Settings
from core.models import (MetricsSnapshot, QueueState, Severity, SensorHealth,
                         SensorReading)
from core.simulation import FakeClock, make_reading


class TestBacklog(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.load()
        self.clock = FakeClock()
        self.detector = BacklogDetector(self.settings, clock=self.clock)

    def test_no_backlog_when_flow_healthy(self):
        verdict = self.detector.update(46, 0.92, 12.0)
        self.assertFalse(verdict.active)

    def test_no_backlog_when_queue_small(self):
        verdict = self.detector.update(5, 0.90, 2.0)
        self.assertFalse(verdict.active)

    def test_backlog_requires_persistence(self):
        for i in range(4):
            verdict = self.detector.update(46, 0.92, 2.0)
            self.assertFalse(verdict.active, f"frame {i}")
            self.clock.advance(2.0)
        verdict = self.detector.update(46, 0.92, 2.0)  # 8s elapsed, need 10s
        self.assertFalse(verdict.active)
        self.clock.advance(2.0)
        verdict = self.detector.update(46, 0.92, 2.0)  # 10s elapsed
        self.assertTrue(verdict.active)
        self.assertIsNotNone(verdict.since)

    def test_backlog_clears_when_condition_drops(self):
        for _ in range(6):
            self.detector.update(46, 0.92, 2.0)
            self.clock.advance(2.0)
        self.assertTrue(self.detector.update(46, 0.92, 2.0).active)
        self.clock.advance(2.0)
        self.assertFalse(self.detector.update(5, 0.92, 2.0).active)


class TestAlertEngine(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.load()
        self.engine = AlertEngine(self.settings)
        self.metrics = MetricsSnapshot(timestamp="t", people_count=40, density=0.8)

    def _evaluate(self, state, sensors=None, disagreement=(), backlog=False, prev=None):
        return self.engine.evaluate(
            state=state,
            previous_state=prev or QueueState.NORMAL,
            sensors=sensors or {},
            disagreement=list(disagreement),
            backlog_active=backlog,
            confidence=1.0,
            metrics=self.metrics,
        )

    def test_congestion_alert_opens_and_resolves(self):
        alerts = self._evaluate(QueueState.CONGESTED, prev=QueueState.NORMAL)
        types = {a.type for a in alerts}
        self.assertIn("CONGESTION", types)

        # same state again: dedupe, no duplicate
        alerts2 = self._evaluate(QueueState.CONGESTED, prev=QueueState.CONGESTED)
        self.assertEqual(sum(1 for a in alerts2 if a.type == "CONGESTION"), 1)

        # resolved on recovery
        alerts3 = self._evaluate(QueueState.NORMAL, prev=QueueState.CONGESTED)
        self.assertEqual([a for a in alerts3 if a.type == "CONGESTION"], [])

    def test_backlog_alert_deduplicated(self):
        self._evaluate(QueueState.BACKLOG, backlog=True, prev=QueueState.CONGESTED)
        alerts = self._evaluate(QueueState.BACKLOG, backlog=True, prev=QueueState.BACKLOG)
        self.assertEqual(sum(1 for a in alerts if a.type == "BACKLOG"), 1)

    def test_disagreement_alert_opens_then_resolves(self):
        alerts = self._evaluate(QueueState.SENSOR_DISAGREEMENT,
                                disagreement=[("people_count", "a", "b", 45, 2)],
                                prev=QueueState.NORMAL)
        self.assertIn("SENSOR_DISAGREEMENT", {a.type for a in alerts})
        alerts2 = self._evaluate(QueueState.NORMAL, prev=QueueState.SENSOR_DISAGREEMENT)
        self.assertNotIn("SENSOR_DISAGREEMENT", {a.type for a in alerts2})

    def test_sensor_offline_alert(self):
        clock = FakeClock()
        reading = make_reading(clock, "SENSOR_01", "vision",
                               10, 0.2, 0.0, SensorHealth.OFFLINE)
        alerts = self._evaluate(QueueState.NORMAL, sensors={"SENSOR_01": reading})
        self.assertIn("SENSOR_OFFLINE:SENSOR_01", {a.type for a in alerts})

    def test_low_confidence_blocks_alert(self):
        self.settings.alerts.min_confidence = 0.9
        alerts = self.engine.evaluate(
            state=QueueState.CONGESTED, previous_state=QueueState.NORMAL,
            sensors={}, disagreement=[], backlog_active=False,
            confidence=0.5, metrics=self.metrics)
        self.assertEqual(alerts, [])

    def test_acknowledge_lifecycle(self):
        self._evaluate(QueueState.CONGESTED, prev=QueueState.NORMAL)
        alert = self.engine.acknowledge("CONGESTION", by="operator")
        self.assertIsNotNone(alert)
        assert alert is not None
        self.assertEqual(alert.status, AlertStatus.ACKNOWLEDGED)
        self.assertEqual(alert.acknowledged_by, "operator")
    def test_resolved_alert_keeps_history(self):
        self._evaluate(QueueState.CONGESTED, prev=QueueState.NORMAL)
        self._evaluate(QueueState.NORMAL, prev=QueueState.CONGESTED)
        resolved = [a for a in self.engine.history if a.type == "CONGESTION"]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].status, AlertStatus.RESOLVED)
        self.assertIsNotNone(resolved[0].resolved_at)


if __name__ == "__main__":
    unittest.main()
