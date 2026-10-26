"""Tests for camera source selection / rotation controls.

Covers the in-process control signal, the atomic camera.yaml store, and
the three API endpoints against a temporary ``camera.yaml`` /
``mappings.yaml`` — production ``config/`` files are never touched.

Guarantees under test:
- an invalid or unavailable target never writes anything;
- a valid switch persists ``camera.source`` and pends a control action;
- rotation rotates the active mapping (or the legacy pixel geometry),
  persists ``camera.rotation`` atomically, and pends a rotate action;
- a no-op rotation applies nothing.

Written as plain unittest (the ``make test`` gate uses unittest
discover), so no pytest fixtures are involved.
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import yaml

from api_server import app, _camera_probe_cache
import camera_config_store
import camera_control
import detect_people
from mapping import load_mapping


from fastapi.testclient import TestClient  # noqa: E402  (after sys.path setup)

client = TestClient(app)

CAMERA_YAML = """
camera:
  source: "0"
  width: 800
  height: 450
  camera_id: "CAMERA 01"
  roi:
    - [100, 100]
    - [700, 100]
    - [700, 400]
    - [100, 400]
  counting_line:
    - [100, 380]
    - [700, 380]
  xray_side_point: [400, 420]
source:
  shape_w: 800
  shape_h: 450
"""

MAPPINGS_YAML = """
schema_version: 1
cameras:
  CAMERA 01:
    camera_id: CAMERA 01
    schema_version: 1
    roi:
      - [0.125, 0.222222]
      - [0.875, 0.222222]
      - [0.875, 0.888889]
      - [0.125, 0.888889]
    lines:
      in: {points: [[0.875, 0.844444], [0.125, 0.844444]], direction: IN}
      out: {points: [[0.125, 0.844444], [0.875, 0.844444]], direction: OUT}
    reference_point: bottom_center
    source_shape: {width: 800, height: 450}
"""


def _read_yaml(path):
    return yaml.safe_load(Path(path).read_text()) or {}


class _TempEnvTestCase(unittest.TestCase):
    """Common plumbing: temp config dir + stubbed probes/holder."""

    AVAILABLE_INDEXES = (0, 2)
    RUNNING_INDEX = None

    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="camera-controls-test-")
        self.addCleanup(__import__("shutil").rmtree, tmp, ignore_errors=True)
        self.camera_path = Path(tmp) / "camera.yaml"
        self.camera_path.write_text(CAMERA_YAML, encoding="utf-8")
        self.mappings_path = Path(tmp) / "mappings.yaml"
        self.running_index = self.RUNNING_INDEX

        self._patches = [
            patch.object(detect_people, "DEFAULT_CAMERA_CONFIG_PATH", self.camera_path),
            patch.object(
                detect_people, "probe_camera_index",
                lambda idx, **kw: (idx in self.AVAILABLE_INDEXES, (1280, 720)),
            ),
            patch.object(camera_config_store, "load_camera_yaml",
                         lambda path: _read_yaml(path)),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

        holder_patcher = patch("api_server.get_holder")
        mock_holder = holder_patcher.start()
        self.addCleanup(holder_patcher.stop)
        snap = None
        if self.running_index is not None:
            snap = type("Snap", (), {"camera_state": type(
                "Cam", (), {"source_index": self.running_index})()})()
        mock_holder.return_value.get_latest_snapshot.return_value = snap

        _camera_probe_cache["ts"] = 0.0
        _camera_probe_cache["available"] = {}
        self.addCleanup(camera_control.consume_camera_control)


class TestCameraSignal(unittest.TestCase):
    def test_request_then_consume_delivers_payload(self):
        camera_control.request_camera_control({"action": "switch", "source": "1"}, "api")
        action, payload, source = camera_control.consume_camera_control()
        self.assertEqual(action, "switch")
        self.assertEqual(payload, {"action": "switch", "source": "1"})
        self.assertEqual(source, "api")
        self.assertEqual(camera_control.consume_camera_control(), ("", {}, ""))

    def test_newer_request_replaces_pending(self):
        camera_control.request_camera_control({"action": "rotate", "rotation": 90})
        camera_control.request_camera_control({"action": "switch", "source": "2"})
        action, payload, _ = camera_control.consume_camera_control()
        self.assertEqual((action, payload.get("source")), ("switch", "2"))
        self.assertEqual(camera_control.consume_camera_control(), ("", {}, ""))


class TestCameraYamlStore(unittest.TestCase):
    def _setup(self, tmp: Path):
        cam = tmp / "camera.yaml"
        cam.write_text(CAMERA_YAML, encoding="utf-8")
        return cam

    def test_update_persists_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = self._setup(Path(tmp))
            backup = camera_config_store.update_camera_yaml(
                cam, {"camera.source": "2", "camera.rotation": 90})
            self.assertTrue(Path(str(backup)).exists())
            doc = _read_yaml(cam)
            self.assertEqual(doc["camera"]["source"], "2")
            self.assertEqual(doc["camera"]["rotation"], 90)
            self.assertEqual(doc["camera"]["width"], 800)
            self.assertEqual(doc["source"]["shape_w"], 800)
            self.assertEqual(_read_yaml(backup)["camera"]["source"], "0")

    def test_update_creates_nested_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = self._setup(Path(tmp))
            camera_config_store.update_camera_yaml(cam, {"a.b.c": 3})
            self.assertEqual(_read_yaml(cam)["a"]["b"]["c"], 3)

    def test_missing_file_raises_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                camera_config_store.update_camera_yaml(
                    Path(tmp) / "nope.yaml", {"camera.source": "1"})

    def test_scalar_update_preserves_comments(self):
        original = (
            "# header comment\n"
            "camera:\n"
            "  # operator note about the source\n"
            "  source: \"0\"\n"
            "\n"
            "  width: 800\n"
            "  # layout calibration comment\n"
            "  counting_line:\n"
            "    - [100, 380]\n"
            "source:\n"
            "  shape_w: 800  # trailing comment\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            cam = Path(tmp) / "camera.yaml"
            cam.write_text(original, encoding="utf-8")
            camera_config_store.update_camera_yaml(cam, {"camera.rotation": 90})
            text = cam.read_text(encoding="utf-8")
            self.assertIn("# header comment", text)
            self.assertIn("# operator note about the source", text)
            self.assertIn("# layout calibration comment", text)
            self.assertIn("# trailing comment", text)
            doc = _read_yaml(cam)
            self.assertEqual(doc["camera"]["rotation"], 90)
            self.assertEqual(doc["camera"]["counting_line"], [[100, 380]])
            self.assertEqual(_read_yaml(cam), camera_config_store.load_camera_yaml(cam))

    def test_scalar_update_inserts_missing_key_keeping_comments(self):
        original = (
            "camera:\n"
            "  source: \"0\"\n"
            "  width: 800\n"
            "source:\n"
            "  shape_w: 800\n"
            "# tail comment\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            cam = Path(tmp) / "camera.yaml"
            cam.write_text(original, encoding="utf-8")
            camera_config_store.update_camera_yaml(cam, {"camera.rotation": 180})
            text = cam.read_text(encoding="utf-8")
            self.assertIn("# tail comment", text)
            self.assertEqual(_read_yaml(cam)["camera"]["rotation"], 180)

    def test_legacy_list_update_falls_back_to_full_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = self._setup(Path(tmp))
            camera_config_store.update_camera_yaml(
                cam, {"camera.rotation": 90, "camera.roi": [[1, 2], [3, 4]]})
            doc = _read_yaml(cam)
            self.assertEqual(doc["camera"]["roi"], [[1, 2], [3, 4]])
            self.assertEqual(doc["camera"]["rotation"], 90)


class TestGetSources(_TempEnvTestCase):
    RUNNING_INDEX = 2

    def test_lists_indexes_and_marks_running(self):
        resp = client.get("/api/dashboard/camera/sources")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["current"], 0)
        by_idx = {s["index"]: s for s in data["sources"]}
        self.assertIs(by_idx[0]["available"], True)
        self.assertIs(by_idx[1]["available"], False)
        self.assertIs(by_idx[2]["available"], True)   # running → no probe needed
        self.assertIs(by_idx[2]["running"], True)


class TestSelectEndpoint(_TempEnvTestCase):
    def test_select_persists_and_signals(self):
        resp = client.post("/api/dashboard/camera/select", json={"index": 2})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIs(data["applied"], True)
        self.assertEqual(data["source_index"], 2)
        self.assertEqual(_read_yaml(self.camera_path)["camera"]["source"], "2")
        action, payload, _ = camera_control.consume_camera_control()
        self.assertEqual((action, payload.get("source")), ("switch", "2"))

    def test_select_unavailable_changes_nothing(self):
        before = _read_yaml(self.camera_path)
        resp = client.post("/api/dashboard/camera/select", json={"index": 1})
        data = resp.json()
        self.assertIs(data["applied"], False)
        self.assertIn("not available", data["errors"][0])
        self.assertEqual(_read_yaml(self.camera_path), before)
        self.assertEqual(camera_control.consume_camera_control(), ("", {}, ""))

    def test_select_running_index_is_a_noop_success(self):
        self.RUNNING_INDEX = 2
        # Subclass-level fixture trick does not apply per-method; run the
        # scenario directly with the running index patched on the holder.
        # The snapshot must look like a LIVE camera (link ONLINE): that is
        # the only case where "already running" is a truthful no-op.
        with patch("api_server.get_holder") as mock_holder:
            snap = type("Snap", (), {"camera_state": type(
                "Cam", (), {"source_index": 2, "link": "ONLINE"})()})()
            mock_holder.return_value.get_latest_snapshot.return_value = snap
            resp = client.post("/api/dashboard/camera/select", json={"index": 2})
        data = resp.json()
        self.assertIs(data["applied"], True)
        self.assertEqual(camera_control.consume_camera_control(), ("", {}, ""))

    def test_select_running_index_retries_when_camera_offline(self):
        """With the camera offline, re-selecting it is the manual retry.

        The no-op shortcut must not swallow the request: a switch to the
        same index has to reach the loop so it re-opens the device.
        """
        with patch("api_server.get_holder") as mock_holder, \
                patch("detect_people.probe_camera_index", return_value=(True, (640, 480))):
            snap = type("Snap", (), {"camera_state": type(
                "Cam", (), {"source_index": 2, "link": "OFFLINE"})()})()
            mock_holder.return_value.get_latest_snapshot.return_value = snap
            resp = client.post("/api/dashboard/camera/select", json={"index": 2})
        data = resp.json()
        self.assertIs(data["applied"], True)
        action, payload, _src = camera_control.consume_camera_control()
        self.assertEqual(action, "switch")
        self.assertEqual(payload.get("source"), "2")

    def test_select_out_of_range_rejected(self):
        resp = client.post("/api/dashboard/camera/select", json={"index": 99})
        self.assertEqual(resp.status_code, 200)
        self.assertIs(resp.json()["applied"], False)


class TestRotationEndpoint(_TempEnvTestCase):
    def _with_mapping(self):
        self.mappings_path.write_text(MAPPINGS_YAML, encoding="utf-8")

    def test_rotate_mapping_transforms_and_persists(self):
        self._with_mapping()
        resp = client.post("/api/dashboard/camera/rotation", json={"degrees": 90})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIs(data["applied"], True)
        self.assertEqual(data["rotation_deg"], 90)
        rotated = load_mapping(self.mappings_path, camera_id="CAMERA 01")
        x, y = rotated.roi[0]
        self.assertAlmostEqual(x, 1.0 - 0.222222, places=4)  # CW90: (1-y, x)
        self.assertAlmostEqual(y, 0.125, places=4)
        self.assertEqual(rotated.source_shape, (450, 800))
        self.assertEqual(_read_yaml(self.camera_path)["camera"]["rotation"], 90)
        action, payload, _ = camera_control.consume_camera_control()
        self.assertEqual((action, payload.get("rotation")), ("rotate", 90))
        self.assertTrue(Path(str(self.mappings_path) + ".bak").exists())

    def test_invalid_degree_rejected_without_writes(self):
        before = _read_yaml(self.camera_path)
        resp = client.post("/api/dashboard/camera/rotation", json={"degrees": 45})
        self.assertIs(resp.json()["applied"], False)
        self.assertEqual(_read_yaml(self.camera_path), before)
        self.assertEqual(camera_control.consume_camera_control(), ("", {}, ""))

    def test_noop_rotation_applies_nothing(self):
        resp = client.post("/api/dashboard/camera/rotation", json={"degrees": 90})
        self.assertIs(resp.json()["applied"], True)
        camera_control.consume_camera_control()
        resp = client.post("/api/dashboard/camera/rotation", json={"degrees": 90})
        self.assertIs(resp.json()["applied"], True)
        self.assertEqual(camera_control.consume_camera_control(), ("", {}, ""))

    def test_rotate_legacy_pixel_geometry(self):
        """No mappings.yaml: legacy camera.yaml pixels rotate in place."""
        resp = client.post("/api/dashboard/camera/rotation", json={"degrees": 180})
        self.assertIs(resp.json()["applied"], True)
        doc = _read_yaml(self.camera_path)
        self.assertEqual(doc["camera"]["rotation"], 180)
        self.assertEqual(doc["camera"]["roi"][0], [800 - 100, 450 - 100])
        self.assertEqual(doc["source"]["shape_w"], 800)
        self.assertEqual(doc["source"]["shape_h"], 450)
        # The legacy-derived mapping stays in-memory: no mappings.yaml appears.
        self.assertFalse(self.mappings_path.exists())


if __name__ == "__main__":
    unittest.main()
