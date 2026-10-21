"""Unit tests for the camera-only queue analytics (app/queue_prototype.py).

Synthetic tracks only — no camera, no YOLO. The webcam is never opened.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from queue_prototype import (
    BACKLOG,
    CONGESTED,
    CROWDED,
    EMPTY,
    NORMAL,
    SPARSE,
    UNCONFIGURED,
    BACKLOG_UNAVAILABLE_MSG,
    CameraConfig,
    DashboardSeries,
    QueueAnalytics,
    SystemMetrics,
    apply_layout_to_config,
    build_dashboard_model,
    draw_dashboard,
    draw_overlay,
    load_camera_config,
    preview_layout,
    recommendation_for,
    _scale_coords,
)
from mapping import LineMapping, MappingGeometry

ROI = [[50, 50], [450, 50], [450, 350], [50, 350]]
LINE = [[450, 50], [450, 350]]
SIDE = (600, 200)  # X-ray side is to the right of the line
SHAPE = (450, 800)  # (h, w)


def make_mapping(reference_point="bottom_center"):
    """A mapping for the interactive-calibration tests.

    Frame 800 x 450. ROI band x in [200, 600] px, y in [180, 405] px.
    Both counting lines are vertical, ordered bottom→top, so the
    positive half-plane of each is to the RIGHT of the line: a forward
    crossing is a rightward walk (IN at x=200, OUT at x=600 px).
    """
    return MappingGeometry(
        camera_id="CAMERA 01",
        roi=((0.25, 0.4), (0.75, 0.4), (0.75, 0.9), (0.25, 0.9)),
        lines={
            "in": LineMapping(points=((0.25, 0.9), (0.25, 0.1)), direction="IN"),
            "out": LineMapping(points=((0.75, 0.9), (0.75, 0.1)), direction="OUT"),
        },
        reference_point=reference_point,
        source_shape=(800, 450),
    )



def make_config(tmpdir, **overrides):
    path = os.path.join(tmpdir, "camera.yaml")
    cfg = CameraConfig()
    cfg.roi = ROI
    cfg.counting_line = LINE
    cfg.xray_side_point = SIDE
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.path = path
    return cfg


def track(tid, x1, y1, x2, y2):
    return [x1, y1, x2, y2, tid]


class TestConfig(unittest.TestCase):
    def test_defaults_are_unconfigured(self):
        cfg = CameraConfig()
        self.assertFalse(cfg.roi_configured)
        self.assertFalse(cfg.line_configured)
        self.assertFalse(cfg.calibrated)
        self.assertFalse(cfg.baseline_available)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "camera.yaml")
            apply_layout_to_config(path, ROI, LINE, SIDE, SHAPE[1], SHAPE[0])
            cfg = load_camera_config(path)
            self.assertTrue(cfg.roi_configured)
            self.assertTrue(cfg.line_configured)
            self.assertEqual(cfg.queue_capacity, 30)
            self.assertEqual(cfg.source_shape_w, SHAPE[1])
            self.assertEqual(cfg.source_shape_h, SHAPE[0])

    def test_validation_rejects_short_roi(self):
        cfg = CameraConfig(roi=[[0, 0], [10, 10]])
        self.assertIn("at least 3", cfg.validate_roi()[0])

    def test_validation_rejects_degenerate_roi(self):
        cfg = CameraConfig(roi=[[0, 0], [10, 0], [20, 0]])
        self.assertIn("degenerate", cfg.validate_roi()[0])

    def test_validation_rejects_out_of_frame(self):
        cfg = CameraConfig(roi=[[0, 0], [10, 0], [0, 10], [99999, 5]])
        self.assertTrue(any("outside the frame" in e for e in cfg.validate_roi(SHAPE)))

    def test_baseline_null_means_backlog_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            self.assertFalse(cfg.baseline_available)


class TestOccupancy(unittest.TestCase):
    def test_counts_only_people_inside_roi(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        tracks = [
            track(1, 100, 100, 140, 180),   # inside
            track(2, 200, 120, 240, 200),   # inside
            track(3, 650, 100, 690, 180),   # outside (right of line)
        ]
        snap = a.update(tracks, SHAPE, now_unix=1000.0)
        self.assertEqual(snap["people_in_queue"], 2)
        self.assertEqual(snap["people_detected"], 3)

    def test_occupancy_zero_when_roi_not_configured(self):
        cfg = CameraConfig()
        a = QueueAnalytics(cfg)
        snap = a.update([track(1, 100, 100, 140, 180)], SHAPE, now_unix=1000.0)
        self.assertEqual(snap["state"], UNCONFIGURED)
        self.assertIn("roi_not_configured", snap["reasons"])


class TestLineCrossing(unittest.TestCase):
    def test_moving_toward_xray_is_exit(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        # person walks left->right across x=450 line (toward X-ray side)
        # 4 frames: one extra hold frame on the far side, because
        # `crossing_debounce_frames` (default 2) requires the track to be
        # observed on the new side for 2 consecutive frames before a
        # crossing is committed. Same business assertion as before.
        for i, cx in enumerate([100, 300, 500, 500]):
            snap = a.update(
                [track(1, cx - 20, 150, cx + 20, 200)],
                SHAPE, now_unix=1000.0 + i * 0.1,
            )
        exits = [e for e in snap["events_since_last"] if e.get("event_type") == "line_crossing"]
        self.assertTrue(exits)
        self.assertEqual(exits[-1]["direction"], "EXIT")
        self.assertEqual(snap["flow"]["current_per_min"], 1)

    def test_moving_away_from_xray_is_enter(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        # walk right->left across x=450 line (away from X-ray side)
        # crossing happens between frame 1 (right) and frame 2 (left);
        # one extra hold frame for the default 2-frame crossing debounce.
        a.update([track(2, 480, 150, 520, 200)], SHAPE, now_unix=2000.0)
        a.update([track(2, 280, 150, 320, 200)], SHAPE, now_unix=2000.1)
        snap = a.update([track(2, 280, 150, 320, 200)], SHAPE, now_unix=2000.2)
        crossings = [e for e in snap["events_since_last"] if e.get("event_type") == "line_crossing"]
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[-1]["direction"], "ENTER")
        self.assertEqual(snap["flow"]["current_per_min"], 0)

    def test_exit_requires_prev_centroid(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        # single frame, person already past the line: no crossing event
        snap = a.update([track(1, 520, 150, 560, 200)], SHAPE, now_unix=3000.0)
        self.assertFalse(any(e.get("event_type") == "line_crossing" for e in snap["events_since_last"]))


class TestWaitingTime(unittest.TestCase):
    def test_waiting_grows_and_resets_on_exit(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        now = 4000.0
        for i in range(10):
            snap = a.update(
                [track(1, 100, 150, 140, 200)], SHAPE, now_unix=now + i,  # inside ROI, waiting
            )
            now += 1
        self.assertGreaterEqual(snap["waiting"]["average_s"], 9.0)
        self.assertGreaterEqual(snap["waiting"]["longest_s"], 9.0)

    def test_waiting_measured_through_exit(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        now = 5000.0
        for i in range(5):  # wait 5s inside
            a.update([track(1, 100, 150, 140, 200)], SHAPE, now_unix=now)
            now += 1
        # 3 exit-side frames so the default 2-frame crossing debounce commits;
        # the event lands on the commit frame, so collect across the run.
        crossing_events = []
        for _ in range(3):
            snap = a.update([track(1, 520, 150, 560, 200)], SHAPE, now_unix=now)  # exit
            now += 1
            crossing_events += [
                e for e in snap["events_since_last"] if e.get("event_type") == "line_crossing"
            ]
        self.assertEqual(crossing_events[-1]["direction"], "EXIT")
        self.assertGreaterEqual(crossing_events[-1]["waiting_s"], 5.0)


class TestSpacing(unittest.TestCase):
    def test_pairwise_pixel_spacing(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        snap = a.update(
            [
                track(1, 100, 100, 140, 180),
                track(2, 200, 120, 240, 200),
                track(3, 300, 110, 340, 190),
            ],
            SHAPE, now_unix=6000.0,
        )
        sp = snap["spacing"]
        self.assertEqual(sp["pairs"], 3)
        self.assertIsNotNone(sp["average_px"])
        self.assertGreater(sp["minimum_px"], 0)

    def test_no_spacing_with_single_person(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        snap = a.update([track(1, 100, 100, 140, 180)], SHAPE, now_unix=6000.0)
        self.assertIsNone(snap["spacing"]["average_px"])

    def test_raw_density_uncalibrated(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        snap = a.update([track(1, 100, 100, 140, 180)], SHAPE, now_unix=6000.0)
        rd = snap["raw_density"]
        self.assertFalse(rd["calibrated"])
        self.assertGreater(rd["people_per_px2"], 0)


class TestClassifier(unittest.TestCase):
    def test_ladder_sparse_normal_crowded(self):
        cfg = make_config(tempfile.mkdtemp(), debounce_frames=1)
        a = QueueAnalytics(cfg)
        now = 7000.0
        cases = {
            EMPTY: 0,
            SPARSE: 4,      # 4/30 = 0.133
            NORMAL: 13,     # 13/30 = 0.433
            CROWDED: 19,    # 19/30 = 0.633
        }
        for expected, n in cases.items():
            # 5-column grid inside the ROI (x < 450, y < 350)
            tracks = [track(i + 1, 60 + (i % 5) * 70, 90 + (i // 5) * 55, 80 + (i % 5) * 70, 130 + (i // 5) * 55)
                      for i in range(n)]
            snap = a.update(tracks, SHAPE, now_unix=now)
            self.assertEqual(snap["state"], expected, f"n={n}")
            now += 1

    def test_congested_requires_low_spacing(self):
        cfg = make_config(tempfile.mkdtemp(), debounce_frames=1,
                          congested_max_spacing_px=30, congested_require_spacing=True)
        a = QueueAnalytics(cfg)
        # 25 people (occupancy 83%), all inside ROI but spread out -> not congested
        tracks = [track(i + 1, 60 + (i % 5) * 90, 80 + (i // 5) * 55, 80 + (i % 5) * 90, 130 + (i // 5) * 55)
                  for i in range(25)]
        snap = a.update(tracks, SHAPE, now_unix=8000.0)
        self.assertEqual(snap["state"], CROWDED)
        self.assertIn("queue_occupancy_above_congested_threshold", snap["reasons"])
        self.assertIn("spacing_above_congested_threshold", snap["reasons"])

    def test_congested_with_tight_spacing(self):
        cfg = make_config(tempfile.mkdtemp(), debounce_frames=1,
                          congested_max_spacing_px=120, congested_require_spacing=True)
        a = QueueAnalytics(cfg)
        # 25 people packed together (occupancy 83%, tight grid)
        tracks = [track(i + 1, 60 + (i % 5) * 30, 80 + (i // 5) * 30, 90 + (i % 5) * 30, 130 + (i // 5) * 30)
                  for i in range(25)]
        snap = a.update(tracks, SHAPE, now_unix=8100.0)
        self.assertEqual(snap["state"], CONGESTED)
        self.assertIn("spacing_below_threshold", snap["reasons"])

    def test_debounce_defers_commit(self):
        cfg = make_config(tempfile.mkdtemp(), debounce_frames=5)
        a = QueueAnalytics(cfg)
        now = 9000.0
        # 3 people -> occupancy 0.10, candidate SPARSE, but commit needs 5 frames
        for i in range(3):
            snap = a.update([track(t, 100 + t * 60, 150, 130 + t * 60, 200) for t in range(3)],
                            SHAPE, now_unix=now)
            now += 1
        self.assertEqual(snap["candidate_state"], SPARSE)
        self.assertNotEqual(snap["state"], SPARSE)

    def test_backlog_unavailable_without_baseline(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        snap = a.update([track(1, 100, 150, 140, 200)], SHAPE, now_unix=10000.0)
        self.assertEqual(snap["backlog"]["check"], "UNAVAILABLE")
        self.assertIn("BACKLOG CHECK UNAVAILABLE", snap["backlog"]["message"])

    def test_backlog_fires_when_baseline_configured(self):
        cfg = make_config(tempfile.mkdtemp(),
                          baseline_people_per_minute=10.0,
                          backlog_persistence_seconds=1.0,
                          backlog_min_people=2,
                          backlog_min_occupancy=0.0,
                          backlog_min_wait_seconds=3.0,
                          debounce_frames=1)
        a = QueueAnalytics(cfg)
        now = 11000.0
        # 4 people waiting inside, no flow (baseline 10/min, low-flow needs < 5/min)
        # wait must reach 3s, then persist 1s -> need ~10 frames at 0.5s
        for _ in range(12):
            tracks = [track(t, 100 + t * 40, 150, 130 + t * 40, 200) for t in range(4)]
            snap = a.update(tracks, SHAPE, now_unix=now)
            now += 0.5
        self.assertEqual(snap["state"], BACKLOG)
        self.assertIn("high_queue_count", snap["reasons"])
        self.assertIn("flow_significantly_below_baseline", snap["reasons"])
        self.assertIn("long_waiting_time", snap["reasons"])
        self.assertIn("condition_sustained", snap["reasons"])


class TestStopwatch(unittest.TestCase):
    def test_start_end_events(self):
        cfg = make_config(tempfile.mkdtemp(), debounce_frames=1,
                          congested_max_spacing_px=500, congested_require_spacing=False)
        a = QueueAnalytics(cfg)
        now = 12000.0
        # 25 packed people -> CONGESTED (spacing confirmation disabled)
        packed = [track(i + 1, 60 + (i % 5) * 30, 80 + (i // 5) * 30, 90 + (i % 5) * 30, 130 + (i // 5) * 30)
                  for i in range(25)]
        snap = a.update(packed, SHAPE, now_unix=now)
        self.assertEqual(snap["state"], CONGESTED)
        self.assertTrue(snap["congestion"]["active"])
        self.assertIsNotNone(snap["congestion"]["started_at"])
        starts = [e for e in snap["events_since_last"] if e.get("event_type") == "congestion_start"]
        self.assertEqual(len(starts), 1)

        # clear the queue -> congestion ends
        snap = a.update([], SHAPE, now_unix=now + 5.0)
        self.assertFalse(snap["congestion"]["active"])
        ends = [e for e in snap["events_since_last"] if e.get("event_type") == "congestion_end"]
        self.assertEqual(len(ends), 1)
        self.assertGreaterEqual(ends[0]["duration_s"], 5.0)


class TestTrackLifecycle(unittest.TestCase):
    def test_lost_track_removed_after_grace(self):
        cfg = make_config(tempfile.mkdtemp(), track_lost_grace_seconds=0.5)
        a = QueueAnalytics(cfg)
        now = 13000.0
        a.update([track(1, 100, 150, 140, 200)], SHAPE, now_unix=now)
        snap = a.update([], SHAPE, now_unix=now + 1.0)  # gone for 1s > grace
        lost = [e for e in snap["events_since_last"] if e.get("event_type") == "track_lost"]
        self.assertEqual(len(lost), 1)
        self.assertEqual(snap["people_in_queue"], 0)


class TestCameraQualityConfig(unittest.TestCase):
    def test_new_fields_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "camera.yaml")
            cfg = make_config(tmp, width=1920, height=1080, display_scale=0.5, detect_mode="head")
            cfg.save(path)
            loaded = load_camera_config(path)
            self.assertEqual(loaded.width, 1920)
            self.assertEqual(loaded.height, 1080)
            self.assertAlmostEqual(loaded.display_scale, 0.5)
            self.assertEqual(loaded.detect_mode, "head")

    def test_detect_mode_defaults_to_person(self):
        cfg = CameraConfig()
        self.assertEqual(cfg.detect_mode, "person")
        self.assertEqual(cfg.width, 0)
        self.assertEqual(cfg.height, 0)
        self.assertAlmostEqual(cfg.display_scale, 1.0)

    def test_detect_mode_normalized_to_lowercase(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "camera.yaml")
            cfg = make_config(tmp, detect_mode="HEAD")
            cfg.save(path)
            loaded = load_camera_config(path)
            self.assertEqual(loaded.detect_mode, "head")

    def test_empty_fields_stay_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "camera.yaml")
            cfg = make_config(tmp, width=0, height=0, display_scale=0.0, detect_mode="")
            cfg.save(path)
            loaded = load_camera_config(path)
            self.assertEqual(loaded.width, 0)
            self.assertEqual(loaded.height, 0)
            self.assertAlmostEqual(loaded.display_scale, 1.0)
            self.assertEqual(loaded.detect_mode, "person")

    def test_source_field_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "camera.yaml")
            cfg = make_config(tmp, source="1")
            cfg.save(path)
            loaded = load_camera_config(path)
            self.assertEqual(loaded.source, "1")

    def test_source_int_normalized_to_str(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "camera.yaml")
            cfg = make_config(tmp)
            cfg.source = 2  # simulate YAML int
            cfg.save(path)
            loaded = load_camera_config(path)
            self.assertEqual(loaded.source, "2")

    def test_source_defaults_to_none(self):
        cfg = CameraConfig()
        self.assertIsNone(cfg.source)


class TestOverlaySmoke(unittest.TestCase):
    """HUD drawing must never crash, with and without a configured layout."""

    def _snapshot(self, state):
        return {
            "state": state,
            "reasons": ["example_reason"],
            "people_in_queue": 3,
            "people_detected": 3,
            "queue_occupancy_percent": 40.0,
            "config_status": {"roi_configured": True, "line_configured": True, "calibrated": False},
            "flow": {"current_per_min": 5.0, "avg_5min_per_min": 4.2},
            "waiting": {"average_s": 63.5, "longest_s": 120.0, "active": 3},
            "spacing": {"average_px": 55.0, "minimum_px": 30.0, "median_px": 50.0, "pairs": 2},
            "raw_density": {"people_per_px2": 1e-5, "calibrated": False},
            "congestion": {"active": False, "duration_s": 0.0, "started_at": None, "ended_at": None},
            "backlog": {"check": "UNAVAILABLE", "fire": False},
        }

    def test_overlay_draws_with_configured_roi(self):
        cfg = make_config(tempfile.mkdtemp())
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        snap = self._snapshot(NORMAL)
        out = draw_overlay(frame, snap, [track(7, 100, 100, 140, 180)], cfg,
                           fps=24.0, track_confs={7: 0.91}, inf_ms=12.0, debug=True)
        self.assertEqual(out.shape, frame.shape)

    def test_overlay_draws_without_roi(self):
        cfg = CameraConfig()
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        out = draw_overlay(frame, self._snapshot(UNCONFIGURED), [], cfg)
        self.assertEqual(out.shape, frame.shape)

    def test_overlay_draws_backlog_state(self):
        cfg = make_config(tempfile.mkdtemp())
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        snap = self._snapshot(BACKLOG)
        snap["backlog"] = {"check": "AVAILABLE", "fire": True}
        snap["congestion"] = {"active": True, "duration_s": 300.0, "started_at": "2026-08-18T00:00:00Z", "ended_at": None}
        out = draw_overlay(frame, snap, [track(1, 50, 60, 90, 120)], cfg,
                           fps=30.0, track_confs={1: 0.95})
        self.assertEqual(out.shape, frame.shape)

    def test_preview_layout_draws(self):
        cfg = make_config(tempfile.mkdtemp())
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        out = preview_layout(frame, cfg)
        self.assertEqual(out.shape, frame.shape)


class TestOverlayCoordinateScaling(TestOverlaySmoke):
    """draw_overlay maps native coords into display space via display_scale."""

    def test_identity_scale_preserves_legacy_render(self):
        cfg = make_config(tempfile.mkdtemp())
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        snap = self._snapshot(NORMAL)
        tracks = [track(7, 100, 100, 140, 180)]
        # display_scale=1.0 must render exactly like the legacy (no-arg) call.
        legacy = draw_overlay(frame, snap, tracks, cfg, fps=24.0,
                              track_confs={7: 0.91}, inf_ms=12.0, debug=True)
        scaled = draw_overlay(frame, snap, tracks, cfg, fps=24.0,
                              track_confs={7: 0.91}, inf_ms=12.0, debug=True,
                              display_scale=1.0)
        self.assertTrue(np.array_equal(legacy, scaled))
        self.assertEqual(scaled.shape, frame.shape)

    def test_half_scale_draws_on_given_display_frame(self):
        cfg = make_config(tempfile.mkdtemp())
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        snap = self._snapshot(NORMAL)
        tracks = [track(7, 100, 100, 140, 180)]
        out = draw_overlay(frame, snap, tracks, cfg, display_scale=0.5)
        self.assertEqual(out.shape, frame.shape)

    def test_half_scale_moves_geometry_to_scaled_coordinates(self):
        cfg = make_config(tempfile.mkdtemp())
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        snap = self._snapshot(NORMAL)
        tracks = [track(7, 100, 100, 140, 180)]
        full = draw_overlay(frame, snap, tracks, cfg, display_scale=1.0)
        half = draw_overlay(frame, snap, tracks, cfg, display_scale=0.5)
        # Scaled layout is drawn at half-size coordinates, so pixels must differ.
        self.assertFalse(np.array_equal(full, half))

    def test_scale_coords_helper(self):
        pts = [[100, 200], [50, 60]]
        self.assertEqual(_scale_coords(pts, 1.0), pts)  # identity, no copy needed
        scaled = _scale_coords(pts, 0.5)
        self.assertEqual(scaled, [[50.0, 100.0], [25.0, 30.0]])


class TestEntryExitTotals(unittest.TestCase):
    def test_enter_and_exit_totals(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        # ENTER: right->left across x=450 (away from X-ray side)
        a.update([track(1, 480, 150, 520, 200)], SHAPE, now_unix=14000.0)
        a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=14000.1)
        snap = a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=14000.2)
        self.assertEqual(snap["total_entries"], 1)
        self.assertEqual(snap["total_exits"], 0)
        # EXIT: left->right (toward X-ray side)
        snap = a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=14000.2)
        snap = a.update([track(1, 480, 150, 520, 200)], SHAPE, now_unix=14000.3)
        snap = a.update([track(1, 480, 150, 520, 200)], SHAPE, now_unix=14000.4)
        self.assertEqual(snap["total_entries"], 1)
        self.assertEqual(snap["total_exits"], 1)

    def test_crossing_counted_once(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        a.update([track(1, 480, 150, 520, 200)], SHAPE, now_unix=15000.0)
        a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=15000.1)
        # standing still after the crossing must not add a second entry
        snap = a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=15000.2)
        snap = a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=15000.3)
        self.assertEqual(snap["total_entries"], 1)
        self.assertEqual(snap["total_exits"], 0)


class TestEntryFlowAndMovement(unittest.TestCase):
    def test_entry_rate_populated(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        a.update([track(1, 480, 150, 520, 200)], SHAPE, now_unix=16000.0)
        a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=16000.1)
        snap = a.update([track(1, 280, 150, 320, 200)], SHAPE, now_unix=16000.2)
        self.assertEqual(snap["flow"]["entry_current_per_min"], 1)
        self.assertEqual(snap["flow"]["current_per_min"], 0)

    def test_movement_lifecycle(self):
        cfg = make_config(tempfile.mkdtemp())
        a = QueueAnalytics(cfg)
        snap = a.update([], SHAPE, now_unix=17000.0)  # empty queue
        self.assertEqual(snap["movement"], "IDLE")
        # a person waits inside ROI with no crossings -> STAGNANT
        snap = a.update([track(1, 100, 150, 140, 200)], SHAPE, now_unix=17001.0)
        self.assertEqual(snap["movement"], "STAGNANT")
        # a second person exits while the first stays -> MOVING (queue non-empty)
        a.update([track(1, 100, 150, 140, 200), track(2, 100, 150, 140, 200)], SHAPE, now_unix=17002.0)
        a.update([track(1, 100, 150, 140, 200), track(2, 480, 150, 520, 200)], SHAPE, now_unix=17003.0)
        snap = a.update(
            [track(1, 100, 150, 140, 200), track(2, 480, 150, 520, 200)], SHAPE, now_unix=17004.0
        )
        self.assertEqual(snap["movement"], "MOVING")


class TestStateChangeAndDuration(unittest.TestCase):
    def test_state_change_event_emitted(self):
        cfg = make_config(tempfile.mkdtemp(), debounce_frames=1)
        a = QueueAnalytics(cfg)
        snap = a.update([], SHAPE, now_unix=18000.0)  # EMPTY, initialization
        changes = [e for e in snap["events_since_last"] if e.get("event_type") == "state_change"]
        self.assertEqual(changes, [])  # no transition yet
        tracks = [track(i + 1, 60 + (i % 5) * 70, 90 + (i // 5) * 55, 80 + (i % 5) * 70, 130 + (i // 5) * 55)
                  for i in range(4)]
        snap = a.update(tracks, SHAPE, now_unix=18001.0)
        changes = [e for e in snap["events_since_last"] if e.get("event_type") == "state_change"]
        self.assertTrue(changes)
        self.assertEqual((changes[-1]["from_state"], changes[-1]["to_state"]), (EMPTY, SPARSE))

    def test_state_duration_grows(self):
        cfg = make_config(tempfile.mkdtemp(), debounce_frames=1)
        a = QueueAnalytics(cfg)
        a.update([], SHAPE, now_unix=19000.0)
        snap = a.update([], SHAPE, now_unix=19005.0)
        self.assertGreaterEqual(snap["state_duration_s"], 5.0)


class TestMetricsBuilder(unittest.TestCase):
    def _snapshot(self, state):
        return {
            "state": state,
            "reasons": ["example_reason"],
            "people_in_queue": 3,
            "people_detected": 4,
            "queue_occupancy_percent": 40.0,
            "occupancy_avg": {"window_s": 120.0, "people": 3.0, "percent": 10.0},
            "total_entries": 12,
            "total_exits": 9,
            "movement": "STAGNANT",
            "state_duration_s": 30.0,
            "config_status": {"roi_configured": True, "line_configured": True, "calibrated": False},
            "flow": {"current_per_min": 2.0, "entry_current_per_min": 1.5},
            "waiting": {"average_s": 63.5, "longest_s": 120.0, "active": 3},
            "spacing": {"average_px": 55.0, "minimum_px": 30.0, "median_px": 50.0, "pairs": 2},
            "congestion": {"active": False, "duration_s": 0.0, "started_at": None, "ended_at": None},
            "backlog": {"check": "UNAVAILABLE", "fire": False},
        }

    def test_builder_maps_snapshot_fields(self):
        cfg = make_config(tempfile.mkdtemp())
        sysm = SystemMetrics(fps=24.0, inference_ms=34.0, tracking_ms=1.2, analytics_ms=0.3,
                             render_ms=6.0, capture_fps=30.0, frame_count=100,
                             detections=4, tracks=3, source="WEBCAM")
        m = build_dashboard_model(self._snapshot(CROWDED), sysm, cfg)
        q = m.queue
        self.assertEqual(q.state, CROWDED)
        self.assertEqual(q.people_in_queue, 3)
        self.assertEqual(q.occupancy_percent, 40.0)
        self.assertEqual(q.total_entries, 12)
        self.assertEqual(q.total_exits, 9)
        self.assertEqual(q.movement, "STAGNANT")
        self.assertEqual(q.entry_flow, 1.5)
        self.assertEqual(q.exit_flow, 2.0)
        self.assertEqual(q.calibration, "PX ONLY")
        self.assertIn("lane", q.recommendation.lower())
        self.assertEqual(m.system.fps, 24.0)
        self.assertEqual(m.system.source, "WEBCAM")

    def test_unconfigured_occupancy_hidden(self):
        cfg = make_config(tempfile.mkdtemp())
        snap = self._snapshot(UNCONFIGURED)
        snap["config_status"] = {"roi_configured": False, "line_configured": False, "calibrated": False}
        m = build_dashboard_model(snap, SystemMetrics(), cfg)
        self.assertIsNone(m.queue.occupancy_percent)
        self.assertEqual(m.queue.recommendation, recommendation_for(UNCONFIGURED, cfg))

    def test_calibration_badge(self):
        cfg = make_config(tempfile.mkdtemp(), px_per_metre=10.0)
        m = build_dashboard_model(self._snapshot(NORMAL), SystemMetrics(), cfg)
        self.assertEqual(m.queue.calibration, "METRES")


class TestDashboardSmoke(unittest.TestCase):
    def _model(self, state=NORMAL, configured=True, events=None):
        snap = {
            "state": state,
            "reasons": ["example_reason"],
            "people_in_queue": 3,
            "people_detected": 3,
            "queue_occupancy_percent": 40.0,
            "occupancy_avg": {"window_s": 120.0, "people": 3.0, "percent": 10.0},
            "total_entries": 12,
            "total_exits": 9,
            "movement": "MOVING",
            "state_duration_s": 30.0,
            "config_status": {"roi_configured": configured, "line_configured": configured, "calibrated": False},
            "flow": {"current_per_min": 2.0, "entry_current_per_min": 1.5},
            "waiting": {"average_s": 63.5, "longest_s": 120.0, "active": 3},
            "spacing": {"average_px": 55.0, "minimum_px": 30.0, "median_px": 50.0, "pairs": 2},
            "congestion": {"active": True, "duration_s": 305.0, "started_at": "2026-08-20T00:00:00Z", "ended_at": None},
            "backlog": {"check": "UNAVAILABLE", "fire": False},
        }
        sysm = SystemMetrics(fps=24.0, inference_ms=34.0, tracking_ms=1.2, analytics_ms=0.3,
                             render_ms=6.0, capture_fps=30.0, frame_count=100,
                             detections=3, tracks=3, source="WEBCAM")
        return build_dashboard_model(snap, sysm, make_config(tempfile.mkdtemp()), recent_events=events)

    def test_dashboard_renders_configured(self):
        cfg = make_config(tempfile.mkdtemp())
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        model = self._model(NORMAL, events=[{
            "event_type": "line_crossing", "direction": "ENTER", "track_id": 3,
            "event_time": "2026-08-20T00:00:00Z",
        }])
        series = DashboardSeries(120)
        for i in range(10):
            series.append(i, model.queue)
        out = draw_dashboard(frame, model, cfg, [track(7, 100, 100, 140, 180)],
                             track_confs={7: 0.91}, frame_index=10, series=series)
        self.assertGreater(out.shape[1], frame.shape[1])  # right panel adds width
        self.assertGreater(out.shape[0], frame.shape[0])  # header + charts add height

    def test_dashboard_renders_unconfigured(self):
        cfg = CameraConfig()
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        model = self._model(UNCONFIGURED, configured=False)
        out = draw_dashboard(frame, model, cfg, [])
        self.assertGreater(out.shape[0], frame.shape[0])
        self.assertGreater(out.shape[1], frame.shape[1])

    def test_series_buffer_caps_history(self):
        buf = DashboardSeries(history=5)
        model = self._model()
        for i in range(10):
            buf.append(i, model.queue)
        self.assertEqual(len(buf.entry_flow.values()), 5)
        self.assertEqual(buf.entry_flow.values()[-1], (9.0, 1.5))


class TestDashboardConfigRoundTrip(unittest.TestCase):
    def test_new_fields_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "camera.yaml")
            cfg = make_config(tmp, camera_id="CAM 02", movement_window_s=90.0, movement_min_events=2,
                              occupancy_avg_window_s=180.0, dashboard_panel_width=420,
                              dashboard_chart_history=200, recommendations={CONGESTED: "X"})
            cfg.save(path)
            loaded = load_camera_config(path)
            self.assertEqual(loaded.camera_id, "CAM 02")
            self.assertEqual(loaded.movement_window_s, 90.0)
            self.assertEqual(loaded.movement_min_events, 2)
            self.assertEqual(loaded.occupancy_avg_window_s, 180.0)
            self.assertEqual(loaded.dashboard_panel_width, 420)
            self.assertEqual(loaded.dashboard_chart_history, 200)
            self.assertEqual(loaded.recommendations.get(CONGESTED), "X")


class TestMappingDrivenCrossings(unittest.TestCase):
    """Mapping is the single active geometry source (replaces legacy)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cfg = make_config(self.tmp)
        cfg.mapping = make_mapping()  # valid mapping from mappings.yaml
        cfg.reference_point = None  # mapping's own value decides
        self.a = QueueAnalytics(cfg)

    def _crossings(self, snap, *types):
        return [
            e for e in snap["events_since_last"]
            if e.get("event_type") in types or (
                types == ("all",) and e.get("event_type", "").startswith("line_")
            )
        ]

    def test_in_crossing_registers_entry_and_keeps_occupancy(self):
        """Regression: an IN crossing must NOT finalise the exit state.

        The pre-mapping code set ``exit_unix``/``total_waiting_s`` on any
        crossing, so an ENTER event evicted the person from the queue
        filter (occ. filter uses ``exit_unix is None``) — occupancy
        dropped although the passenger had just joined.
        """
        now = 8000.0
        # left of the IN line (x=200 px), outside ROI
        self.a.update([track(1, 100, 190, 140, 250)], SHAPE, now_unix=now)
        now += 0.1
        # crosses IN rightwards; one hold frame for the 2-frame debounce
        snap = self.a.update([track(1, 240, 190, 280, 250)], SHAPE, now_unix=now)
        now += 0.1
        snap = self.a.update([track(1, 240, 190, 280, 250)], SHAPE, now_unix=now)
        ins = [
            e for e in snap["events_since_last"]
            if e["event_type"] == "line_crossing"
            and e.get("line_direction") == "IN" and not e.get("reversed")
        ]
        self.assertEqual(len(ins), 1)
        self.assertEqual(snap["total_entries"], 1)
        self.assertEqual(snap["total_exits"], 0)
        ps = self.a.persons.get(1)
        self.assertIsNotNone(ps)
        self.assertIsNone(ps.exit_unix, "IN crossing must not finalise the exit")
        self.assertIsNone(ps.total_waiting_s)
        # occupancy = live ROI membership, NOT entries-minus-exits
        self.assertEqual(snap["people_in_queue"], 1)
        # more frames inside: occupancy must stay 1
        now += 0.1
        snap = self.a.update([track(1, 240, 190, 280, 250)], SHAPE, now_unix=now)
        self.assertEqual(snap["people_in_queue"], 1)

    def test_out_crossing_finalises_waiting_and_exit(self):
        now = 8100.0
        # enter through the IN line
        self.a.update([track(1, 240, 190, 280, 250)], SHAPE, now_unix=now)
        now += 0.1
        self.a.update([track(1, 240, 190, 280, 250)], SHAPE, now_unix=now)
        now += 1.5  # wait a moment inside
        # walk right across the OUT line (x=600 px); commit needs a hold
        snap = self.a.update([track(1, 620, 190, 660, 250)], SHAPE, now_unix=now)
        now += 0.1
        snap = self.a.update([track(1, 620, 190, 660, 250)], SHAPE, now_unix=now)
        outs = [
            e for e in snap["events_since_last"]
            if e["event_type"] == "line_crossing"
            and e.get("line_direction") == "OUT" and not e.get("reversed")
        ]
        self.assertEqual(len(outs), 1)
        self.assertEqual(snap["total_exits"], 1)
        self.assertEqual(snap["total_entries"], 0 if snap["people_in_queue"] else 0)
        ps = self.a.persons.get(1)
        self.assertIsNotNone(ps)
        self.assertIsNotNone(ps.exit_unix)
        self.assertIsNotNone(ps.total_waiting_s)
        self.assertIn("waiting_s", outs[0])
        # after leaving, occupancy drops to 0
        now += 0.1
        snap = self.a.update([track(1, 620, 190, 660, 250)], SHAPE, now_unix=now)
        self.assertEqual(snap["people_in_queue"], 0)

    def test_legacy_geometry_ignored_while_mapping_valid(self):
        """A valid mapping must be the single geometry source.

        The legacy camera.yaml geometry in ``make_config`` points at a
        completely different area; crossing there must produce nothing.
        """
        now = 8200.0
        # legacy line is x=450: walking 520→300 would cross it under the
        # legacy code. The mapping's IN line is x=200, OUT at x=600.
        self.a.update([track(1, 520, 190, 560, 250)], SHAPE, now_unix=now)
        now += 0.1
        snap = self.a.update([track(1, 300, 190, 340, 250)], SHAPE, now_unix=now)
        now += 0.1
        snap = self.a.update([track(1, 300, 190, 340, 250)], SHAPE, now_unix=now)
        line_events = [
            e for e in snap["events_since_last"] if e["event_type"].startswith("line_")
        ]
        self.assertEqual(line_events, [])

    def test_roi_membership_uses_bottom_center_reference(self):
        """bbox bottom inside the ROI ⇒ inside, even when centroid isn't.

        ROI top edge is y=180 px. A box (300,140)-(340,200) has centroid
        y=170 (outside under legacy centroid rules) but bottom-center
        y=200 (inside under the mapping reference).
        """
        now = 8300.0
        snap = self.a.update([track(5, 300, 140, 340, 200)], SHAPE, now_unix=now)
        self.assertEqual(snap["people_in_queue"], 1)
        self.assertTrue(self.a.persons[5].inside_roi)

    def test_centroid_reference_fallback_reproduces_legacy_semantics(self):
        cfg = make_config(self.tmp)
        cfg.mapping = make_mapping(reference_point="centroid")
        a = QueueAnalytics(cfg)
        now = 8400.0
        snap = a.update([track(5, 300, 140, 340, 200)], SHAPE, now_unix=now)
        # centroid y=170 is above the ROI top edge (180) → not inside
        self.assertEqual(snap["people_in_queue"], 0)
        self.assertFalse(a.persons[5].inside_roi)

    def test_occupancy_independent_of_entries_and_exits(self):
        """Occupancy is the live ROI member count, never a difference."""
        now = 8500.0
        # two people inside, no crossings yet
        self.a.update([track(1, 240, 190, 280, 250), track(2, 400, 190, 440, 250)], SHAPE, now_unix=now)
        now += 0.1
        snap = self.a.update(
            [track(1, 240, 190, 280, 250), track(2, 400, 190, 440, 250)], SHAPE, now_unix=now
        )
        self.assertEqual(snap["people_in_queue"], 2)
        # person 2 crosses OUT (two frames to commit)
        now += 0.1
        self.a.update([track(1, 240, 190, 280, 250), track(2, 620, 190, 660, 250)], SHAPE, now_unix=now)
        now += 0.1
        snap = self.a.update(
            [track(1, 240, 190, 280, 250), track(2, 620, 190, 660, 250)], SHAPE, now_unix=now
        )
        # 1 person still inside — not entries(1 or 0) minus exits arithmetic
        self.assertEqual(snap["people_in_queue"], 1)
        self.assertEqual(snap["total_exits"], 1)

    def test_jitter_at_line_produces_single_event(self):
        """Debounce: a 1-frame flip to the far side must NOT count.

        IN line at x=200; far side is x>200. Frames alternate across the
        line but never hold 2 consecutive frames on the far side.
        """
        cfg = make_config(self.tmp)
        cfg.mapping = make_mapping()
        a = QueueAnalytics(cfg)
        now = 8600.0
        positions = [190, 210, 190, 210, 190, 210, 190]
        total_events = 0
        for cx in positions:
            snap = a.update([track(7, cx - 20, 190, cx + 20, 250)], SHAPE, now_unix=now)
            now += 0.1
            total_events += len(
                [e for e in snap["events_since_last"] if e["event_type"].startswith("line_")]
            )
        self.assertEqual(total_events, 0)

    def test_crossing_commits_after_hold_frames(self):
        cfg = make_config(self.tmp)
        cfg.mapping = make_mapping()
        a = QueueAnalytics(cfg)
        now = 8700.0
        events = []
        for cx in [190, 210, 210]:  # hold 2 consecutive frames on far side
            snap = a.update([track(7, cx - 20, 190, cx + 20, 250)], SHAPE, now_unix=now)
            now += 0.1
            events += [e for e in snap["events_since_last"] if e["event_type"].startswith("line_")]
        self.assertEqual(len(events), 1)

    def test_reverse_crossing_is_explicit_and_uncounted(self):
        """Crossing back out of the positive side fires a reversed event
        and never changes the directional counters."""
        cfg = make_config(self.tmp)
        cfg.mapping = make_mapping()
        a = QueueAnalytics(cfg)
        now = 8800.0
        seq = [190, 210, 210, 190, 190]  # in, hold, back out, hold
        seen = []
        for cx in seq:
            snap = a.update([track(9, cx - 20, 190, cx + 20, 250)], SHAPE, now_unix=now)
            now += 0.1
            seen += [e for e in snap["events_since_last"] if e["event_type"].startswith("line_")]
        forward = [e for e in seen if not e.get("reversed")]
        reversed_ = [e for e in seen if e.get("reversed")]
        self.assertEqual(len(forward), 1)
        self.assertGreaterEqual(len(reversed_), 1)
        self.assertEqual(reversed_[0]["event_type"], "line_crossing_reversed")
        self.assertEqual(reversed_[0]["direction"], "ENTER")
        snap = a.update([], SHAPE, now_unix=now)
        self.assertEqual(snap["total_entries"], 1, "reverse crossing must not add a second IN")

    def test_mapping_override_survives_debounce_config(self):
        """crossing_debounce_frames is honoured per configuration."""
        cfg = make_config(self.tmp)
        cfg.mapping = make_mapping()
        cfg.crossing_debounce_frames = 1  # legacy single-frame behaviour
        a = QueueAnalytics(cfg)
        now = 8900.0
        snap = a.update([track(3, 140, 190, 180, 250)], SHAPE, now_unix=now)
        now += 0.1
        snap = a.update([track(3, 190, 190, 230, 250)], SHAPE, now_unix=now)
        events = [e for e in snap["events_since_last"] if e["event_type"].startswith("line_")]
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
