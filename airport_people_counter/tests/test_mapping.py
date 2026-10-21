"""Tests for the mapping schema + geometry module (`app/mapping.py`).

Covers what the interactive camera mapping features depends on:
normalised-coordinate round trips, validation rules, the legacy
camera.yaml conversion (both horizontal and vertical counting lines),
and the atomic persistence with backup.
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from mapping import (
    BOTTOM_CENTER_REFERENCE,
    CENTROID_REFERENCE,
    LineMapping,
    MappingError,
    MappingGeometry,
    PixelGeometry,
    default_mappings_path,
    from_legacy_camera_config,
    from_pixel,
    is_simple_polygon,
    load_mapping,
    reference_xy,
    save_mapping,
    to_pixel,
    validate_mapping,
)


def _valid_mapping(**overrides) -> MappingGeometry:
    roi = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]
    lines = {
        "in": LineMapping(points=((0.9, 0.85), (0.1, 0.85)), direction="IN"),
        "out": LineMapping(points=((0.1, 0.85), (0.9, 0.85)), direction="OUT"),
    }
    m = MappingGeometry(camera_id="CAMERA 01", roi=tuple(map(tuple, roi)), lines=dict(lines))
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


class TestCoordinateConversions(unittest.TestCase):
    def test_to_pixel_scales_by_frame_shape(self):
        m = _valid_mapping()
        pix = to_pixel(m, (1280, 720))
        self.assertEqual(pix.roi[0], (0.1 * 1280, 0.1 * 720))
        self.assertEqual(pix.in_line, ((0.9 * 1280, 0.85 * 720), (0.1 * 1280, 0.85 * 720)))
        self.assertEqual(pix.in_direction, "IN")
        self.assertEqual(pix.out_direction, "OUT")
        self.assertEqual(pix.reference_point, BOTTOM_CENTER_REFERENCE)

    def test_from_pixel_is_inverse_of_to_pixel(self):
        m = _valid_mapping(source_shape=(1000, 500))
        pix = to_pixel(m, (1000, 500))
        back = from_pixel(
            pix.roi,
            {"in": (pix.in_line[0], pix.in_line[1], "IN"),
             "out": (pix.out_line[0], pix.out_line[1], "OUT")},
            shape=(1000, 500),
        )
        for original, rebuilt in zip(m.roi, back.roi):
            self.assertAlmostEqual(original[0], rebuilt[0], places=6)
            self.assertAlmostEqual(original[1], rebuilt[1], places=6)
        self.assertAlmostEqual(
            back.lines["out"].points[1][0], m.lines["out"].points[1][0], places=6
        )

    def test_to_pixel_rejects_zero_shape(self):
        with self.assertRaises(MappingError):
            to_pixel(_valid_mapping(), (0, 0))

    def test_reference_xy_bottom_center(self):
        # bbox (100, 100, 300, 300): bottom-center is (200, 300).
        self.assertEqual(reference_xy((100, 100, 300, 300), BOTTOM_CENTER_REFERENCE), (200.0, 300.0))
        # centroid is (200, 200).
        self.assertEqual(reference_xy((100, 100, 300, 300), CENTROID_REFERENCE), (200.0, 200.0))


class TestValidation(unittest.TestCase):
    def test_valid_mapping_has_no_errors(self):
        self.assertEqual(validate_mapping(_valid_mapping(), shape=(1280, 720)), [])

    def test_roi_too_few_vertices(self):
        m = _valid_mapping()
        m.roi = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.8))
        # triangle is fine…
        self.assertEqual(validate_mapping(m), [])
        m.roi = ((0.1, 0.1), (0.9, 0.1))
        errors = validate_mapping(m)
        self.assertTrue(any("at least 3 vertices" in e for e in errors))

    def test_coordinates_out_of_bounds(self):
        m = _valid_mapping()
        m.roi = ((0.1, 0.1), (0.9, 0.1), (0.9, 1.2), (0.1, 0.8))
        errors = validate_mapping(m)
        self.assertTrue(any("#2" in e and "outside" in e for e in errors))

    def test_degenerate_roi_zero_area(self):
        m = _valid_mapping()
        m.roi = ((0.1, 0.1), (0.4, 0.1), (0.7, 0.1), (0.9, 0.1))
        errors = validate_mapping(m)
        self.assertTrue(any("zero area" in e for e in errors))

    def test_self_intersecting_roi(self):
        m = _valid_mapping()
        # bowtie polygon
        m.roi = ((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9))
        self.assertFalse(is_simple_polygon(m.roi))
        errors = validate_mapping(m)
        self.assertTrue(any("self-intersects" in e for e in errors))

    def test_line_degenerate_identical_endpoints(self):
        m = _valid_mapping()
        m.lines["in"] = LineMapping(points=((0.5, 0.5), (0.5, 0.5)), direction="IN")
        errors = validate_mapping(m, shape=(1280, 720))
        self.assertTrue(any("identical" in e for e in errors))

    def test_line_too_short_in_pixels(self):
        m = _valid_mapping()
        # 0.0005 * diag(1280,720)≈1469 px → ~0.73 px, far below the
        # 1%-of-diagonal floor (≈14.7 px).
        m.lines["in"] = LineMapping(points=((0.5, 0.5), (0.5005, 0.5)), direction="IN")
        errors = validate_mapping(m, shape=(1280, 720))
        self.assertTrue(any("too short" in e for e in errors))

    def test_identical_in_and_out_lines(self):
        m = _valid_mapping()
        m.lines["in"] = LineMapping(points=((0.1, 0.85), (0.9, 0.85)), direction="OUT")
        errors = validate_mapping(m)
        self.assertTrue(any("identical" in e and "IN and OUT" in e for e in errors))

    def test_unsupported_direction(self):
        m = _valid_mapping()
        m.lines["in"] = LineMapping(points=((0.9, 0.85), (0.1, 0.85)), direction="SIDEWAYS")
        errors = validate_mapping(m)
        self.assertTrue(any("direction" in e for e in errors))

    def test_unsupported_reference_point(self):
        m = _valid_mapping(reference_point="knee")
        errors = validate_mapping(m)
        self.assertTrue(any("reference_point" in e for e in errors))

    def test_unsupported_schema_version(self):
        m = _valid_mapping(schema_version=99)
        errors = validate_mapping(m)
        self.assertTrue(any("schema_version" in e for e in errors))

    def test_reverse_direction_lines_are_allowed(self):
        """Same geometry, opposite directions islegitimate (two waypoints)."""
        m = _valid_mapping()
        m.lines["out"] = LineMapping(points=((0.1, 0.85), (0.9, 0.85)), direction="OUT")
        m.lines["in"] = LineMapping(points=((0.9, 0.85), (0.1, 0.85)), direction="IN")
        self.assertEqual(validate_mapping(m), [])


class TestLegacyConversion(unittest.TestCase):
    def test_horizontal_line_positive_side_is_xray_side(self):
        """camera.yaml layout: horizontal line, xray BELOW (y larger)."""
        m = from_legacy_camera_config(
            roi=[[160, 100], [1120, 100], [1120, 520], [160, 520]],
            counting_line=[[160, 560], [1120, 560]],
            xray_side_point=[640, 650],
            shape=(1280, 720),
        )
        out = m.lines["out"]
        inl = m.lines["in"]
        # out line's positive half-plane must contain the xray side.
        from mapping import _line_side_px
        self.assertGreater(_line_side_px(out.points[0], out.points[1], 640 / 1280, 650 / 720), 0)
        self.assertLess(_line_side_px(inl.points[0], inl.points[1], 640 / 1280, 650 / 720), 0)
        self.assertEqual(m.reference_point, CENTROID_REFERENCE)
        # roi normalised
        self.assertAlmostEqual(m.roi[0][0], 160 / 1280, places=9)

    def test_vertical_line_positive_side_is_xray_side(self):
        """Legacy test-suite layout: vertical line, xray to the RIGHT >450."""
        m = from_legacy_camera_config(
            roi=[[50, 50], [450, 50], [450, 350], [50, 350]],
            counting_line=[[450, 50], [450, 350]],
            xray_side_point=[600, 200],
            shape=(800, 450),
        )
        from mapping import _line_side_px
        out = m.lines["out"]
        # Positive half-plane must include a point right of the line.
        self.assertGreater(_line_side_px(out.points[0], out.points[1], 600 / 800, 200 / 450), 0)
        # A point left of the line must map to the IN line's positive side.
        self.assertGreater(_line_side_px(m.lines["in"].points[0], m.lines["in"].points[1], 100 / 800, 200 / 450), 0)


class TestPersistence(unittest.TestCase):
    def test_save_and_reload_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mappings.yaml"
            m = _valid_mapping(source_shape=(1280, 720))
            save_mapping(path, m)
            loaded = load_mapping(path, camera_id="CAMERA 01")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.lines["out"].points, m.lines["out"].points)
            self.assertEqual(loaded.lines["out"].direction, "OUT")
            self.assertEqual(loaded.reference_point, BOTTOM_CENTER_REFERENCE)
            self.assertAlmostEqual(loaded.roi[2][0], 0.9, places=9)

    def test_atomic_write_leaves_backup_and_no_tmp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mappings.yaml"
            save_mapping(path, _valid_mapping())
            first = path.read_text()
            save_mapping(path, _valid_mapping(camera_id="CAMERA 01"))
            backup = Path(str(path) + ".bak")
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(), first)
            leftovers = [p.name for p in Path(tmp).iterdir() if p.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_missing_camera_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mappings.yaml"
            save_mapping(path, _valid_mapping())
            self.assertIsNone(load_mapping(path, camera_id="CAMERA 99"))

    def test_corrupt_file_raises_usable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mappings.yaml"
            path.write_text("cameras: [not, a, mapping]\n")
            with self.assertRaises(MappingError):
                load_mapping(path, camera_id="CAMERA 01")

    def test_default_mappings_path_beside_camera_yaml(self):
        self.assertEqual(
            default_mappings_path(Path("/x/config/camera.yaml")),
            Path("/x/config/mappings.yaml"),
        )
        self.assertEqual(
            default_mappings_path(Path("/x/config")),
            Path("/x/config/mappings.yaml"),
        )


class TestSchemaRoundTrip(unittest.TestCase):
    def test_from_dict_accepts_dict_and_list_line_forms(self):
        dict_form = {
            "schema_version": 1,
            "camera_id": "CAMERA 01",
            "roi": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]],
            "lines": {
                "in": {"points": [[0.9, 0.85], [0.1, 0.85]], "direction": "IN"},
                "out": {"points": [[0.1, 0.85], [0.9, 0.85]], "direction": "OUT"},
            },
            "reference_point": "bottom_center",
        }
        m = MappingGeometry.from_dict(dict_form)
        self.assertEqual(sorted(m.lines), ["in", "out"])

        list_form = dict(dict_form)
        list_form["lines"] = [
            {"id": "in", "points": [[0.9, 0.85], [0.1, 0.85]], "direction": "IN"},
            {"id": "out", "points": [[0.1, 0.85], [0.9, 0.85]], "direction": "OUT"},
        ]
        m2 = MappingGeometry.from_dict(list_form)
        self.assertEqual(sorted(m2.lines), ["in", "out"])

    def test_from_dict_rejects_bad_direction(self):
        with self.assertRaises(MappingError):
            MappingGeometry.from_dict({
                "schema_version": 1,
                "roi": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8]],
                "lines": {
                    "in": {"points": [[0.9, 0.85], [0.1, 0.85]], "direction": "UP"},
                },
            })


if __name__ == "__main__":
    unittest.main()
