"""
Regression tests for the defects found in the audit of 2026-10-20.

Each test here corresponds to a verified defect, so a future change that
reintroduces the defect fails loudly:

* a video-file source reported the camera as OFFLINE while detection was
  working (the delivered frame rate was only measured for live sources);
* the frame age was a fabricated constant, so LIVE / STALE / OFFLINE was
  decided by an assumption rather than a measurement;
* metrics that the pipeline does not measure were reported as 0.0, which
  is indistinguishable from a real measurement of zero;
* the dashboard's occupancy kept updating while the queue was frozen
  offline, so the two numbers disagreed;
* analytics events had no operator-facing label and fell back to a raw
  event-type string;
* ``/api/shutdown`` called ``os._exit(0)``, which skipped every cleanup
  step (camera release, buffered writes, in-flight requests).

The tests use temporary directories and never touch real monitoring data.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
for path in (str(PROJECT_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from queue_prototype import CameraConfig, QueueAnalytics, SystemMetrics  # noqa: E402
from shutdown import (  # noqa: E402
    request_shutdown,
    reset_shutdown,
    shutdown_reason,
    shutdown_requested,
    wait_for_shutdown,
)
from snapshot import CameraState, ConfigStatus, QueueSnapshot, SnapshotBuilder  # noqa: E402


def make_config(**overrides) -> CameraConfig:
    """A camera config with a usable ROI and counting line."""
    config = CameraConfig(
        width=1280,
        height=720,
        roi=[[100.0, 100.0], [900.0, 100.0], [900.0, 600.0], [100.0, 600.0]],
        counting_line=[[100.0, 650.0], [900.0, 650.0]],
        xray_side_point=[500.0, 700.0],
        queue_capacity=30,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def track(tid: int, x1: float, y1: float, x2: float, y2: float):
    """A SORT-style track row: x1, y1, x2, y2, id."""
    return [x1, y1, x2, y2, tid]


def make_snapshot(**overrides) -> QueueSnapshot:
    """An internal QueueSnapshot with sensible live values."""
    base = dict(
        timestamp="2026-10-20T10:00:00.000Z",
        frame_id=10,
        freshness="LIVE",
        staleness_ms=40,
        fps=25.0,
        inference_ms=21.0,
        tracking_ms=0.3,
        analytics_ms=0.3,
        detection_count=3,
        tracked_people=3,
        people_in_queue=3,
        total_in=10,
        total_out=7,
        queue_state="NORMAL",
        occupancy_percent=10.0,
        entry_rate=1.0,
        exit_rate=1.0,
        avg_wait_s=60.0,
        longest_wait_s=90.0,
        congestion_duration_s=0.0,
        spacing_px=120.0,
        movement="MOVING",
        config_status=ConfigStatus(
            roi_configured=True,
            line_configured=True,
            calibrated=False,
            classifier_enabled=True,
        ),
        calibration="PX ONLY",
        camera_state=CameraState(
            link="ONLINE", fps=25.0, resolution="1280x720", frame_age_ms=40
        ),
    )
    base.update(overrides)
    return QueueSnapshot(**base)


class TestFrameFreshness(unittest.TestCase):
    """The LIVE / STALE / OFFLINE decision must follow a measurement."""

    def setUp(self):
        self.config = make_config()

    def test_no_capture_rate_is_offline(self):
        system = SystemMetrics(capture_fps=0.0)
        state, staleness = SnapshotBuilder._compute_freshness(system, self.config)
        self.assertEqual(state, "OFFLINE")
        self.assertEqual(staleness, 999999)

    def test_fresh_frame_is_live(self):
        system = SystemMetrics(capture_fps=30.0, last_frame_unix=time.time())
        state, staleness = SnapshotBuilder._compute_freshness(system, self.config)
        self.assertEqual(state, "LIVE")
        self.assertGreaterEqual(staleness, 0)
        self.assertLess(staleness, 1000)

    def test_old_frame_is_stale(self):
        # 30 fps -> a frame older than 3 intervals (>100 ms) is stale.
        system = SystemMetrics(capture_fps=30.0, last_frame_unix=time.time() - 1.0)
        state, staleness = SnapshotBuilder._compute_freshness(system, self.config)
        self.assertEqual(state, "STALE")
        self.assertGreater(staleness, 100)

    def test_measured_age_is_reported_not_assumed(self):
        system = SystemMetrics(capture_fps=30.0, last_frame_unix=time.time() - 0.05)
        _, staleness = SnapshotBuilder._compute_freshness(system, self.config)
        # ~50 ms old, so it must not read as the 33 ms one-interval guess.
        self.assertGreater(staleness, 40)

    def test_without_a_timestamp_the_interval_estimate_is_used(self):
        system = SystemMetrics(capture_fps=30.0, last_frame_unix=0.0)
        state, staleness = SnapshotBuilder._compute_freshness(system, self.config)
        self.assertEqual(state, "LIVE")
        self.assertEqual(staleness, 1000 // 30)

    def test_camera_state_follows_the_capture_rate(self):
        system = SystemMetrics(capture_fps=25.0, last_frame_unix=time.time())
        camera = SnapshotBuilder._camera_state(system, self.config)
        self.assertEqual(camera.link, "ONLINE")

    def test_device_reaches_the_snapshot(self):
        from queue_prototype import DashboardModel, QueueMetrics

        # The builder reads the device from the shared metrics model, which
        # is what the runtime passes in (build_dashboard_model uses the same
        # SystemMetrics instance for both).
        system = SystemMetrics(
            capture_fps=25.0, last_frame_unix=time.time(), device="mps", fps=25.0,
        )
        dashboard = DashboardModel(
            queue=QueueMetrics(state="NORMAL", people_in_queue=3, occupancy_percent=10.0),
            system=system,
        )
        snapshot, _frame = SnapshotBuilder.build(dashboard, system, self.config, [], [])
        self.assertEqual(snapshot.device, "mps")


class TestFileSourceFrameRate(unittest.TestCase):
    """A video file delivers frames too -- it must not read as 'no signal'."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlkit-video-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_video(self, frames: int = 12) -> Path:
        import cv2
        import numpy as np

        path = self.tmp / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
        for i in range(frames):
            image = np.full((120, 160, 3), 30, dtype=np.uint8)
            image[20:60, 20 + i * 4: 50 + i * 4] = 220
            writer.write(image)
        writer.release()
        return path

    def test_file_source_reports_a_nonzero_delivered_rate(self):
        from detect_people import FrameStream

        path = self._write_video()
        stream = FrameStream(str(path))
        try:
            stream.start()
            deadline = time.time() + 5.0
            while time.time() < deadline and stream.delivered_fps <= 0:
                time.sleep(0.2)
            self.assertGreater(stream.delivered_fps, 0.0,
                               "a file source must report the rate it delivers")
            self.assertGreater(stream.reported_fps, 0.0,
                               "a file source must report its native rate")
        finally:
            stream.release()

    def test_file_source_records_a_frame_timestamp(self):
        from detect_people import FrameStream

        path = self._write_video()
        stream = FrameStream(str(path))
        try:
            stream.start()
            deadline = time.time() + 5.0
            while time.time() < deadline and stream.last_frame_unix <= 0:
                time.sleep(0.1)
            self.assertGreater(stream.last_frame_unix, 0.0)
            self.assertLess(stream.last_frame_unix, time.time() + 1.0)
        finally:
            stream.release()


class TestHonestMetrics(unittest.TestCase):
    """A value that was never measured must not be reported as zero."""

    def setUp(self):
        from dashboard_adapter import reset_adapter_state

        reset_shutdown()
        self.addCleanup(reset_shutdown)
        reset_adapter_state()
        self.addCleanup(reset_adapter_state)

    def _dashboard(self, snapshot):
        from dashboard_adapter import build_dashboard_snapshot

        return build_dashboard_snapshot(
            snapshot, make_config(), [], server_start_ts=time.time() - 60,
        )

    def test_unmeasured_camera_metrics_are_null(self):
        dashboard = self._dashboard(make_snapshot())
        self.assertIsNone(dashboard.camera.latency_ms)
        self.assertIsNone(dashboard.camera.packet_loss)

    def test_recording_metrics_are_null_when_there_is_no_recorder(self):
        dashboard = self._dashboard(make_snapshot())
        self.assertFalse(dashboard.recording.active)
        self.assertIsNone(dashboard.recording.since_sec)
        self.assertIsNone(dashboard.recording.size_gb)

    def test_device_is_reported_by_the_pipeline(self):
        dashboard = self._dashboard(make_snapshot(device="mps"))
        self.assertEqual(dashboard.system.device, "mps")

    def test_offline_freezes_occupancy_with_the_queue(self):
        online = self._dashboard(make_snapshot())
        self.assertEqual(online.queue.queue, 3)
        self.assertEqual(online.queue.occupancy, 10.0)

        # Now the camera drops out while the numbers keep moving.
        offline = self._dashboard(make_snapshot(
            people_in_queue=0,
            occupancy_percent=0.0,
            camera_state=CameraState(
                link="OFFLINE", fps=0.0, resolution="1280x720", frame_age_ms=999999
            ),
        ))
        self.assertEqual(offline.queue.queue, 3, "the queue reading must be frozen")
        self.assertEqual(offline.queue.occupancy, 10.0,
                         "occupancy must be frozen with the queue, not keep updating")
        self.assertFalse(offline.queue.state_active)


class TestOperatorEventLabels(unittest.TestCase):
    """Events must read like operator notes, not raw event-type strings."""

    def setUp(self):
        self.config = make_config()
        self.analytics = QueueAnalytics(self.config)

    def test_roi_entry_event_has_a_label(self):
        self.analytics.update([track(1, 300, 200, 400, 500)], (720, 1280))
        self.analytics.update([track(1, 300, 200, 400, 500)], (720, 1280))
        events = self.analytics.event_log.events
        self.assertTrue(events, "expected at least one event")
        for event in events:
            self.assertIn("label", event)
            self.assertTrue(event["label"])

    def test_state_change_event_has_a_readable_label(self):
        for _ in range(8):
            self.analytics.update([], (720, 1280))
        events = self.analytics.event_log.events
        state_events = [e for e in events if e.get("event_type") == "state_change"]
        self.assertTrue(state_events)
        self.assertIn("→", state_events[-1]["label"])


class TestCooperativeShutdown(unittest.TestCase):
    """The shutdown request must be observable and resettable."""

    def setUp(self):
        reset_shutdown()
        self.addCleanup(reset_shutdown)

    def test_request_sets_the_flag_and_records_the_reason(self):
        self.assertFalse(shutdown_requested())
        request_shutdown("test")
        self.assertTrue(shutdown_requested())
        self.assertEqual(shutdown_reason(), "test")

    def test_reset_clears_the_flag(self):
        request_shutdown("test")
        reset_shutdown()
        self.assertFalse(shutdown_requested())
        self.assertEqual(shutdown_reason(), "")

    def test_wait_returns_immediately_once_requested(self):
        request_shutdown("test")
        self.assertTrue(wait_for_shutdown(timeout=0.1))

    def test_wait_times_out_while_not_requested(self):
        started = time.time()
        self.assertFalse(wait_for_shutdown(timeout=0.2))
        self.assertGreaterEqual(time.time() - started, 0.15)

    def test_api_shutdown_does_not_kill_the_process(self):
        """The endpoint must request a shutdown, not call os._exit."""
        from api_server import shutdown_server

        calls = []
        original_exit = os._exit
        os._exit = lambda code: calls.append(code)
        try:
            asyncio.run(shutdown_server())
        finally:
            os._exit = original_exit
        self.assertEqual(calls, [], "the endpoint must not call os._exit")
        self.assertTrue(shutdown_requested())


if __name__ == "__main__":
    unittest.main()
