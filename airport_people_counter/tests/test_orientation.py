"""Unit tests for the pure orientation/rotation math (`app/orientation.py`).

A 90°-step rotation of a normalised [0, 1] coordinate plane must be an
exact, aspect-independent transform, and applying it four times must
come back to the identity — these guarantees are what let a rotation
change keep the calibrated region covering the same physical scene.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import numpy as np

from mapping import MappingGeometry, LineMapping, validate_mapping
from orientation import (
    VALID_ROTATIONS,
    delta_rotation,
    is_valid_rotation,
    normalize_rotation,
    rotate_frame,
    rotate_mapping,
    rotate_norm_point,
    rotate_np_rot90,
    rotate_px_point,
    rotated_size,
    transform_legacy_layout,
)


def _valid_mapping() -> MappingGeometry:
    return MappingGeometry(
        camera_id="CAMERA 01",
        roi=((0.1, 0.1), (0.9, 0.1), (0.9, 0.8), (0.1, 0.8)),
        lines={
            "in": LineMapping(points=((0.9, 0.85), (0.1, 0.85)), direction="IN"),
            "out": LineMapping(points=((0.1, 0.85), (0.9, 0.85)), direction="OUT"),
        },
        reference_point="bottom_center",
        source_shape=(1280, 720),
    )


class TestRotationValidation(unittest.TestCase):
    def test_valid_rotations(self):
        self.assertEqual(VALID_ROTATIONS, (0, 90, 180, 270))
        self.assertTrue(is_valid_rotation(90))
        self.assertFalse(is_valid_rotation(45))
        self.assertFalse(is_valid_rotation("x"))

    def test_normalize_rotation_rejects_invalid(self):
        with self.assertRaises(ValueError):
            normalize_rotation(45)
        with self.assertRaises(ValueError):
            normalize_rotation(None)
        self.assertEqual(normalize_rotation("270"), 270)

    def test_delta_rotation_arithmetic(self):
        self.assertEqual(delta_rotation(0, 90), 90)
        self.assertEqual(delta_rotation(90, 0), 270)  # counter-clockwise one step
        self.assertEqual(delta_rotation(180, 180), 0)
        self.assertEqual(delta_rotation(270, 90), 180)


class TestNormalisedPoints(unittest.TestCase):
    def test_known_transforms(self):
        self.assertEqual(rotate_norm_point((0.2, 0.3), 90), (0.7, 0.2))
        self.assertEqual(rotate_norm_point((0.2, 0.3), 180), (0.8, 0.7))
        self.assertEqual(rotate_norm_point((0.2, 0.3), 270), (0.3, 0.8))
        self.assertEqual(rotate_norm_point((0.2, 0.3), 0), (0.2, 0.3))

    def test_corners_map_to_corners(self):
        for deg in VALID_ROTATIONS:
            for p in ((0, 0), (1, 0), (1, 1), (0, 1)):
                u, v = rotate_norm_point(p, deg)
                self.assertIn(u, (0.0, 1.0))
                self.assertIn(v, (0.0, 1.0))


class TestFrameRotation(unittest.TestCase):
    def test_rotate_frame_changes_shape_for_90(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0, 0] = 7
        out = rotate_frame(frame, 90)
        self.assertEqual(out.shape, (3, 2, 3))
        # The original top-left pixel becomes the top-right pixel.
        self.assertEqual(int(out[0, -1, 0]), 7)

    def test_rotate_frame_0_is_identity(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertIs(rotate_frame(frame, 0), frame)

    def test_matches_rot90_reference(self):
        frame = np.random.default_rng(7).integers(0, 255, (4, 6, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(rotate_frame(frame, 90), rotate_np_rot90(frame, 90)))
        self.assertTrue(np.array_equal(rotate_frame(frame, 180), rotate_np_rot90(frame, 180)))
        self.assertTrue(np.array_equal(rotate_frame(frame, 270), rotate_np_rot90(frame, 270)))

    def test_rotated_size(self):
        self.assertEqual(rotated_size(1280, 720, 90), (720, 1280))
        self.assertEqual(rotated_size(1280, 720, 180), (1280, 720))
        self.assertEqual(rotated_size(1280, 720, 0), (1280, 720))


class TestPixelPoints(unittest.TestCase):
    def test_known_pixel_point(self):
        # (10, 10) on a 640x480 frame rotated 90° CW: u = 10/640, v = 10/480
        # → (1 - v, u) × (480, 640) lands exactly on (470, 10).
        x, y = rotate_px_point(10, 10, 90, (640, 480))
        self.assertAlmostEqual(x, 470.0, places=9)
        self.assertAlmostEqual(y, 10.0, places=9)

    def test_identity_for_zero(self):
        self.assertEqual(rotate_px_point(12, 34, 0, (640, 480)), (12.0, 34.0))

    def test_points_cover_the_rotated_frame_only(self):
        pts = [(0.0, 0.0), (639, 0), (639, 479), (0, 479), (320, 240)]
        for deg in VALID_ROTATIONS:
            w, h = rotated_size(640, 480, deg)
            for (x, y) in pts:
                nx, ny = rotate_px_point(x, y, deg, (640, 480))
                self.assertGreaterEqual(nx, -1e-9)
                self.assertGreaterEqual(ny, -1e-9)
                self.assertLessEqual(nx, w)
                self.assertLessEqual(ny, h)

    def test_four_steps_return_the_original_pixel(self):
        for x, y in ((10, 10), (639, 0), (320, 240)):
            px = (float(x), float(y))
            seen = px
            for _ in range(4):
                seen = rotate_px_point(seen[0], seen[1], 90, (640, 480))
            self.assertAlmostEqual(seen[0], px[0], places=6)
            self.assertAlmostEqual(seen[1], px[1], places=6)


class TestLegacyGeometry(unittest.TestCase):
    def test_legacy_layout_transform_mirrors_for_180(self):
        out = transform_legacy_layout(
            roi=[[10, 10], [600, 20], [300, 400]],
            counting_line=[[100, 380], [600, 380]],
            xray_side_point=[400, 420],
            degrees=180,
            size=(640, 480),
        )
        self.assertAlmostEqual(out["roi"][0][0], 640 - 10, places=6)
        self.assertAlmostEqual(out["roi"][0][1], 480 - 10, places=6)
        self.assertAlmostEqual(out["xray_side_point"][0], 640 - 400, places=6)
        self.assertAlmostEqual(out["counting_line"][1][1], 480 - 380, places=6)

    def test_legacy_layout_skips_missing_parts(self):
        out = transform_legacy_layout(
            roi=None, counting_line=None, xray_side_point=None,
            degrees=90, size=(640, 480),
        )
        self.assertEqual(out, {})


class TestMappingTransform(unittest.TestCase):
    def test_rotated_mapping_stays_valid(self):
        m = _valid_mapping()
        for deg in VALID_ROTATIONS:
            rotated = rotate_mapping(m, deg)
            self.assertEqual(validate_mapping(rotated, rotated.source_shape), [],
                             f"rotation {deg} produced an invalid mapping")

    def test_direction_labels_ride_along(self):
        rotated = rotate_mapping(_valid_mapping(), 90)
        self.assertEqual(rotated.lines["in"].direction, "IN")
        self.assertEqual(rotated.lines["out"].direction, "OUT")

    def test_source_shape_swaps_for_90(self):
        rotated = rotate_mapping(_valid_mapping(), 90)
        self.assertEqual(rotated.source_shape, (720, 1280))

    def test_camera_id_and_reference_unchanged(self):
        rotated = rotate_mapping(_valid_mapping(), 270)
        self.assertEqual(rotated.camera_id, "CAMERA 01")
        self.assertEqual(rotated.reference_point, "bottom_center")

    def test_four_steps_return_the_original_mapping(self):
        original = _valid_mapping()
        current = original
        for _ in range(4):
            current = rotate_mapping(current, 90)
        for original_pt, rotated_pt in zip(original.roi, current.roi):
            self.assertAlmostEqual(original_pt[0], rotated_pt[0], places=9)
            self.assertAlmostEqual(original_pt[1], rotated_pt[1], places=9)
        for name in ("in", "out"):
            for original_pt, rotated_pt in zip(original.lines[name].points,
                                               current.lines[name].points):
                self.assertAlmostEqual(original_pt[0], rotated_pt[0], places=9)
                self.assertAlmostEqual(original_pt[1], rotated_pt[1], places=9)


if __name__ == "__main__":
    unittest.main()
