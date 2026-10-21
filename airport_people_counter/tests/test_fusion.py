"""Unit tests for sensor fusion (agreement, disagreement, offline handling)."""

import unittest

from core.config import Settings
from core.fusion import SensorFusion
from core.models import SensorHealth, SensorReading
from core.simulation import FakeClock, make_reading


class TestFusion(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.load()
        self.clock = FakeClock()
        self.fusion = SensorFusion(self.settings, clock=self.clock)

    def _read(self, sid, people, dens, flow, health=SensorHealth.ONLINE):
        return make_reading(self.clock, sid, "vision", people, dens, flow, health)

    def test_agreement_within_tolerance(self):
        self.fusion.ingest(self._read("SENSOR_01", 45, 0.90, 12.0))
        self.fusion.ingest(self._read("SENSOR_02", 46, 0.92, 12.0))
        res = self.fusion.evaluate()
        self.assertEqual(res.disagreements, [])
        self.assertAlmostEqual(res.agreement_confidence, 1.0)

    def test_disagreement_outside_tolerance(self):
        self.fusion.ingest(self._read("SENSOR_01", 45, 0.90, 12.0))
        self.fusion.ingest(self._read("SENSOR_02", 2, 0.04, 12.0))
        res = self.fusion.evaluate()
        metrics = {d[0] for d in res.disagreements}
        self.assertIn("people_count", metrics)
        self.assertIn("density", metrics)
        self.assertLess(res.agreement_confidence, 0.5)

    def test_single_primary_sensor_no_disagreement(self):
        self.fusion.ingest(self._read("SENSOR_01", 15, 0.30, 12.0))
        res = self.fusion.evaluate()
        self.assertEqual(res.disagreements, [])
        self.assertAlmostEqual(res.agreement_confidence, 1.0)
        self.assertTrue(res.primary_ok)

    def test_offline_sensor_excluded(self):
        self.fusion.ingest(self._read("SENSOR_01", 45, 0.90, 12.0))
        self.fusion.ingest(self._read("SENSOR_02", 2, 0.04, 12.0, SensorHealth.OFFLINE))
        res = self.fusion.evaluate()
        # Offline S2 must not cause a disagreement.
        self.assertEqual(res.disagreements, [])
        self.assertTrue(res.primary_ok)

    def test_all_offline_is_primary_error(self):
        self.fusion.ingest(self._read("SENSOR_01", 45, 0.90, 12.0, SensorHealth.OFFLINE))
        self.fusion.ingest(self._read("SENSOR_02", 45, 0.90, 12.0, SensorHealth.OFFLINE))
        res = self.fusion.evaluate()
        self.assertFalse(res.primary_ok)
        self.assertTrue(res.primary_error)

    def test_stale_reading_becomes_offline(self):
        self.fusion.ingest(self._read("SENSOR_01", 45, 0.90, 12.0))
        self.clock.advance(self.settings.sensors.staleness_seconds + 1)
        self.fusion.ingest(self._read("SENSOR_02", 46, 0.92, 12.0))
        res = self.fusion.evaluate()
        self.assertEqual(res.sensors["SENSOR_01"].health, SensorHealth.OFFLINE)
        self.assertTrue(res.primary_ok)  # S2 is still online
        self.assertFalse(res.primary_error)

    def test_downstream_tracks_congestion(self):
        self.fusion.ingest(self._read("SENSOR_01", 45, 0.90, 12.0))
        self.fusion.ingest(self._read("SENSOR_02", 46, 0.92, 12.0))
        self.fusion.ingest(self._read("SENSOR_03", 20, 0.40, 12.0))
        res = self.fusion.evaluate()
        self.assertTrue(res.downstream_ok)
        self.assertFalse(res.downstream_congested)
        # density 0.40 is below crowded(0.60); try higher
        self.fusion.ingest(self._read("SENSOR_03", 35, 0.70, 12.0))
        res = self.fusion.evaluate()
        self.assertTrue(res.downstream_congested)
        # Downstream never causes a people_count disagreement with primary.
        self.assertEqual(res.disagreements, [])

    def test_downstream_unavailable_when_offline(self):
        self.fusion.ingest(self._read("SENSOR_03", 20, 0.40, 12.0, SensorHealth.OFFLINE))
        res = self.fusion.evaluate()
        self.assertFalse(res.downstream_ok)
        self.assertTrue(res.downstream_unavailable)


if __name__ == "__main__":
    unittest.main()
