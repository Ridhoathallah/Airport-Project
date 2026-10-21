"""End-to-end tests: monitor integration + simulation scenarios."""

import unittest

from core.monitor import QueueMonitor
from core.models import (EventType, PassengerEvent, QueueState, SensorHealth,
                         SensorReading)
from core.simulation import (FakeClock, make_reading, run_all)
from core.config import Settings


class TestMonitorIntegration(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.load()
        self.clock = FakeClock()
        self.monitor = QueueMonitor(self.settings, clock=self.clock)

    def _ingest(self, p1=0, d1=0.0, s2p=0, s2d=0.0, flow=12.0):
        self.monitor.ingest_sensor(make_reading(self.clock, "SENSOR_01", "vision",
                                                p1, d1, flow))
        self.monitor.ingest_sensor(make_reading(self.clock, "SENSOR_02", "vision_secondary",
                                                s2p, s2d, flow))
        snap = self.monitor.evaluate()
        self.clock.advance(1.0)
        return snap

    def test_empty_to_congested_pipeline(self):
        snap = self._ingest(0, 0.0, 0, 0.0)
        self.assertEqual(snap.state, QueueState.EMPTY)
        for _ in range(5):
            snap = self._ingest(45, 0.90, 46, 0.92)
        self.assertEqual(snap.state, QueueState.CONGESTED)
        snap = self._ingest(45, 0.90, 46, 0.92)  # one frame into the episode
        self.assertGreater(snap.metrics.congestion_duration_s, 0)
        # CONGESTION alert is open
        types = {a["type"] for a in snap.open_alerts}
        self.assertIn("CONGESTION", types)

    def test_metrics_populated(self):
        snap = self._ingest(20, 0.40, 20, 0.40)
        self.assertAlmostEqual(snap.metrics.people_count, 20)
        self.assertAlmostEqual(snap.metrics.density, 0.40)
        self.assertAlmostEqual(snap.metrics.occupancy_percent, 40.0)

    def test_passenger_events_measure_waiting_time(self):
        self._ingest(1, 0.02, 1, 0.02)
        self.monitor.ingest_passenger_event(PassengerEvent(
            zone_id="XRAY01", event_time=self.clock(), event_type=EventType.ENTER,
            tracking_id="T-1"))
        self.clock.advance(30.0)
        snap = self._ingest(1, 0.02, 1, 0.02)
        self.assertGreater(snap.metrics.average_waiting_time_s, 25.0)

    def test_recovery_clears_congestion_duration(self):
        for _ in range(5):
            self._ingest(45, 0.90, 46, 0.92)
        snap = self._ingest(45, 0.90, 46, 0.92)
        self.assertGreater(snap.metrics.congestion_duration_s, 0)
        for _ in range(6):
            snap = self._ingest(0, 0.0, 0, 0.0)
        self.assertEqual(snap.state, QueueState.EMPTY)
        self.assertEqual(snap.metrics.congestion_duration_s, 0.0)

    def test_sensor_error_when_primary_offline(self):
        self.monitor.ingest_sensor(make_reading(self.clock, "SENSOR_01", "vision",
                                                0, 0.0, 0.0, SensorHealth.OFFLINE))
        self.monitor.ingest_sensor(make_reading(self.clock, "SENSOR_02", "vision_secondary",
                                                0, 0.0, 0.0, SensorHealth.OFFLINE))
        snap = self.monitor.evaluate()
        self.assertEqual(snap.state, QueueState.SENSOR_ERROR)


class TestSimulation(unittest.TestCase):
    def test_all_scenarios_pass(self):
        results = run_all(verbose=False)
        for r in results:
            self.assertTrue(r.passed, f"{r.name}: expected {r.expected} got {r.actual}")
        self.assertEqual(len(results), 11)


if __name__ == "__main__":
    unittest.main()
