"""
Unit tests for the operator-facing event language.

The detector names its states in implementation terms (EMPTY, SPARSE,
CROWDED, UNCONFIGURED). The activity log is read by security staff during
a shift, so it uses plain language, while the engineering description stays
available on the technical field for the diagnostics view.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from snapshot import SnapshotBuilder


def _events(raw):
    return SnapshotBuilder._events_to_api(raw)


def _at(iso):
    return {"event_time": iso, "severity": "info"}


class TestEventLanguage(unittest.TestCase):
    """The activity log speaks to an operator, not to an engineer."""

    def test_track_enter_roi_is_operator_language(self):
        events = _events([{
            "event_type": "track_enter_roi",
            "track_id": 7,
            "event_time": "2026-10-21T10:00:00.000Z",
            "confidence": 0.91,
        }])
        self.assertEqual(events[0].label, "Passenger joined the queue")
        self.assertEqual(events[0].kind, "entry")
        # The engineering detail is kept, but only for the diagnostics view.
        self.assertIn("track_enter_roi", events[0].technical)
        self.assertIn("track_id=7", events[0].technical)

    def test_line_crossing_exit_reports_the_wait(self):
        events = _events([{
            "event_type": "line_crossing",
            "direction": "EXIT",
            "track_id": 3,
            "event_time": "2026-10-21T10:00:00.000Z",
            "waiting_s": 272.0,
        }])
        self.assertEqual(events[0].label, "Passenger cleared screening")
        self.assertEqual(events[0].kind, "exit")
        self.assertEqual(events[0].detail, "Waited 4 min 32 s")

    def test_line_crossing_enter_is_not_called_a_clearing(self):
        events = _events([{
            "event_type": "line_crossing",
            "direction": "ENTER",
            "track_id": 3,
            "event_time": "2026-10-21T10:00:00.000Z",
        }])
        self.assertEqual(events[0].label, "Passenger entered the queue area")
        self.assertEqual(events[0].kind, "entry")

    def test_state_change_never_leaks_an_internal_state_name(self):
        events = _events([{
            "event_type": "state_change",
            "from_state": "UNCONFIGURED",
            "to_state": "EMPTY",
            "event_time": "2026-10-21T10:00:00.000Z",
        }])
        self.assertEqual(events[0].label, "Queue status changed")
        detail = events[0].detail
        self.assertNotIn("UNCONFIGURED", detail)
        self.assertNotIn("EMPTY", detail)
        self.assertEqual(detail, "Setup incomplete to Empty")
        # The raw names stay on the technical field.
        self.assertIn("from_state=UNCONFIGURED", events[0].technical)
        self.assertIn("to_state=EMPTY", events[0].technical)

    def test_state_change_translates_the_crowded_tier(self):
        events = _events([{
            "event_type": "state_change",
            "from_state": "NORMAL",
            "to_state": "CROWDED",
            "event_time": "2026-10-21T10:00:00.000Z",
        }])
        self.assertEqual(events[0].detail, "Normal to Busy")

    def test_congestion_events_are_operator_language(self):
        start = _events([{
            "event_type": "congestion_start",
            "state": "CONGESTED",
            "event_time": "2026-10-21T10:00:00.000Z",
        }])
        self.assertEqual(start[0].label, "Congestion detected")
        self.assertEqual(start[0].kind, "congestion")

        end = _events([{
            "event_type": "congestion_end",
            "event_time": "2026-10-21T10:05:00.000Z",
            "duration_s": 300.0,
        }])
        self.assertEqual(end[0].label, "Queue beginning to clear")
        self.assertEqual(end[0].detail, "Congestion lasted 5 min 0 s")

    def test_track_lost_is_operator_language(self):
        events = _events([{
            "event_type": "track_lost",
            "track_id": 9,
            "event_time": "2026-10-21T10:00:00.000Z",
            "waiting_s": 12.0,
        }])
        self.assertEqual(events[0].label, "Passenger left the camera view")
        self.assertEqual(events[0].kind, "system")

    def test_every_event_carries_a_technical_description(self):
        raw = [
            {"event_type": "track_enter_roi", "track_id": 1,
             "event_time": "2026-10-21T10:00:00.000Z"},
            {"event_type": "line_crossing", "direction": "EXIT", "track_id": 1,
             "event_time": "2026-10-21T10:00:01.000Z", "waiting_s": 4.0},
            {"event_type": "state_change", "from_state": "EMPTY", "to_state": "SPARSE",
             "event_time": "2026-10-21T10:00:02.000Z"},
        ]
        for event in _events(raw):
            self.assertTrue(event.technical)
            self.assertIn(
                event.technical.split(" ")[0],
                ("track_enter_roi", "line_crossing", "state_change"),
            )


class TestCrossingLanguage(unittest.TestCase):
    """A crossing that reverses is still operator language.

    The analytics layer labels the event; the activity log must not show
    the line id, the direction constants or the word "reversed".
    """

    def _cross_back(self):
        """Walk a track across the OUT line and back over it."""
        import tempfile

        from mapping import LineMapping, MappingGeometry
        from queue_prototype import CameraConfig, QueueAnalytics

        cfg = CameraConfig()
        cfg.roi = [[200, 180], [600, 180], [600, 405], [200, 405]]
        cfg.counting_line = [[600, 405], [600, 180]]
        cfg.xray_side_point = [700, 300]
        cfg.mapping = MappingGeometry(
            camera_id="CAMERA 01",
            roi=((0.25, 0.4), (0.75, 0.4), (0.75, 0.9), (0.25, 0.9)),
            lines={
                "in": LineMapping(points=((0.25, 0.9), (0.25, 0.1)), direction="IN"),
                "out": LineMapping(points=((0.75, 0.9), (0.75, 0.1)), direction="OUT"),
            },
            reference_point="bottom_center",
            source_shape=(800, 450),
        )
        cfg.path = os.path.join(tempfile.mkdtemp(), "camera.yaml")
        analytics = QueueAnalytics(cfg)
        shape = (450, 800)

        def track(tid, x1, y1, x2, y2):
            return [x1, y1, x2, y2, tid]

        now = 1000.0
        # approach the OUT line from the left, then cross back to the left
        for cx in (500, 560, 620, 660, 620, 560, 500):
            now += 0.1
            snap = analytics.update([track(1, cx - 20, 200, cx + 20, 260)], shape, now_unix=now)
        return snap["events_since_last"]

    def test_a_reversed_crossing_reads_like_an_operator_event(self):
        events = self._cross_back()
        reversed_events = [e for e in events if e.get("event_type") == "line_crossing_reversed"]
        self.assertTrue(reversed_events, "expected a reversed crossing event")
        for event in reversed_events:
            label = event["label"]
            self.assertNotIn("on in", label)
            self.assertNotIn("on out", label)
            self.assertNotIn("reversed crossing", label.lower())
            self.assertIn("Passenger", label)

    def test_a_reversed_crossing_still_carries_the_engineering_detail(self):
        events = self._cross_back()
        reversed_events = [e for e in events if e.get("event_type") == "line_crossing_reversed"]
        self.assertTrue(reversed_events)
        technical = SnapshotBuilder._technical_event_text(reversed_events[0])
        self.assertIn("line_crossing_reversed", technical)


    def test_camera_switch_and_rotation_speak_operator_language(self):
        events = _events([
            {
                "event_type": "camera_switch",
                "label": "Camera source changed",
                "detail": "CAM-02 selected",
                "event_time": "2026-10-22T09:00:00.000Z",
            },
            {
                "event_type": "camera_rotation",
                "label": "Image rotation changed",
                "detail": "90 degrees clockwise",
                "event_time": "2026-10-22T09:00:01.000Z",
            },
        ])
        self.assertEqual(events[0].label, "Camera source changed")
        self.assertEqual(events[0].detail, "CAM-02 selected")
        self.assertEqual(events[0].kind, "system")
        self.assertEqual(events[1].label, "Image rotation changed")
        self.assertEqual(events[1].detail, "90 degrees clockwise")

    def test_camera_events_name_the_engineering_frame(self):
        events = _events([{
            "event_type": "camera_switch",
            "source": "1",
            "event_time": "2026-10-22T09:00:00.000Z",
        }])
        self.assertIn("camera_switch", events[0].technical)
        self.assertIn("source=1", events[0].technical)


if __name__ == "__main__":
    unittest.main()
