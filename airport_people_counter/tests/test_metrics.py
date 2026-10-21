"""Unit tests for core.metrics (density, occupancy, flow, distance, timing)."""

import unittest

from core.metrics import (density, distance_stats, euclidean_distance,
                          flow_rate, occupancy_percentage, pairwise_distances,
                          percent_below_spacing, processing_time,
                          waiting_time)


class TestMetrics(unittest.TestCase):
    def test_density(self):
        self.assertAlmostEqual(density(20, 50.0), 0.4)
        self.assertEqual(density(20, 0), 0.0)
        self.assertEqual(density(0, 50.0), 0.0)

    def test_occupancy_percentage(self):
        self.assertAlmostEqual(occupancy_percentage(25, 50), 50.0)
        self.assertEqual(occupancy_percentage(10, 0), 0.0)
        self.assertEqual(occupancy_percentage(0, 50), 0.0)

    def test_flow_rate(self):
        self.assertAlmostEqual(flow_rate(120, 60.0), 120.0)  # 2 people/sec -> 120/min
        self.assertAlmostEqual(flow_rate(1, 60.0), 1.0)
        self.assertEqual(flow_rate(5, 0), 0.0)

    def test_waiting_time(self):
        entry = "2024-01-01T00:00:00Z"
        later = "2024-01-01T00:01:30Z"
        self.assertAlmostEqual(waiting_time(entry, later), 90.0)
        self.assertEqual(waiting_time(later, entry), 0.0)  # clock skew never negative

    def test_processing_time(self):
        a = "2024-01-01T00:00:00Z"
        b = "2024-01-01T00:00:45Z"
        self.assertAlmostEqual(processing_time(a, b), 45.0)

    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_distance((0, 0, 0), (3, 4, 0)), 5.0)

    def test_pairwise_distances(self):
        pts = [(0, 0, 0), (3, 4, 0), (0, 0, 6)]
        out = pairwise_distances(pts)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(min(out), 5.0)
        self.assertAlmostEqual(max(out), 7.81025, places=4)

    def test_distance_stats_empty(self):
        stats = distance_stats([])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["mean"], 0.0)

    def test_percent_below_spacing(self):
        self.assertAlmostEqual(percent_below_spacing([0.5, 1.5, 2.0], 1.0), 33.3333, places=2)
        self.assertEqual(percent_below_spacing([], 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
