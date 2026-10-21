"""Integration tests for the interactive mapping API endpoints.

Covers ``GET /api/dashboard/mapping`` and ``PUT /api/dashboard/mapping``
against a temporary ``camera.yaml`` so production ``config/`` files are
never touched: the last known-good mapping must survive an invalid
update, and a valid update must persist atomically and ask the
detection loop to hot-reload.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import pytest
from fastapi.testclient import TestClient

from api_server import app
import detect_people
import reload_signal


client = TestClient(app)

CAMERA_YAML = """
camera:
  source: "test"
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

VALID_PAYLOAD = {
    "schema_version": 1,
    "roi": [[0.125, 0.138889], [0.875, 0.138889], [0.875, 0.722222], [0.125, 0.722222]],
    "lines": {
        "in": {"points": [[0.875, 0.777778], [0.125, 0.777778]], "direction": "IN"},
        "out": {"points": [[0.125, 0.777778], [0.875, 0.777778]], "direction": "OUT"},
    },
    "reference_point": "bottom_center",
    "source_shape": {"width": 1280, "height": 720},
}


@pytest.fixture
def temp_camera_env():
    """A temp directory with camera.yaml, wired into the endpoints."""
    with tempfile.TemporaryDirectory() as tmp:
        camera_path = Path(tmp) / "camera.yaml"
        camera_path.write_text(CAMERA_YAML, encoding="utf-8")
        with patch.object(detect_people, "DEFAULT_CAMERA_CONFIG_PATH", camera_path):
            yield camera_path
    reload_signal.reset_config_reload()


def test_get_mapping_reports_legacy_derived_mapping(temp_camera_env):
    """No mappings.yaml yet: GET reports the in-memory legacy derivation."""
    resp = client.get("/api/dashboard/mapping")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_id"] == "CAMERA 01"
    assert data["mapping_active"] is True
    assert data["mapping_source"] == "legacy"
    assert data["reference_point"] == "centroid"
    assert data["mapping"]["roi"] is not None
    assert {l["id"] for l in data["mapping"]["lines"]} == {"in", "out"}


def test_put_mapping_persists_and_signals_reload(temp_camera_env):
    reload_signal.reset_config_reload()
    resp = client.put("/api/dashboard/mapping", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is True
    assert data["camera_id"] == "CAMERA 01"
    assert data["mapping_active"] is True
    assert data["reference_point"] == "bottom_center"
    assert data["backup_path"] is None  # first write, nothing to back up

    # persisted next to camera.yaml, and a GET now reports file-backed
    mappings_path = temp_camera_env.parent / "mappings.yaml"
    assert mappings_path.exists()
    persisted = json.loads(json.dumps(_load_yaml(mappings_path)))
    assert persisted["cameras"]["CAMERA 01"]["schema_version"] == 1
    assert persisted["cameras"]["CAMERA 01"]["updated_at_iso"]

    resp = client.get("/api/dashboard/mapping")
    data = resp.json()
    assert data["mapping_active"] is True
    assert data["reference_point"] == "bottom_center"

    # the detection loop was asked to reload
    assert reload_signal.consume_config_reload() == "api"


def test_put_mapping_creates_backup_on_update(temp_camera_env):
    client.put("/api/dashboard/mapping", json=VALID_PAYLOAD)
    reload_signal.reset_config_reload()
    resp = client.put("/api/dashboard/mapping", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is True
    assert data["backup_path"] and data["backup_path"].endswith("mappings.yaml.bak")


def test_invalid_mapping_rejected_and_last_known_good_survives(temp_camera_env):
    # establish a good mapping first
    client.put("/api/dashboard/mapping", json=VALID_PAYLOAD)
    reload_signal.reset_config_reload()
    mappings_path = Path(str(temp_camera_env).replace("camera.yaml", "mappings.yaml"))
    good_content = mappings_path.read_text(encoding="utf-8")

    bad = dict(VALID_PAYLOAD)
    bad["roi"] = [[0.5, 0.1], [0.5, 0.1], [0.5, 0.1]]  # degenerate zero-area ROI
    resp = client.put("/api/dashboard/mapping", json=bad)
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is False
    assert data["errors"], "must carry human-readable validation errors"
    # disk untouched: the running pipeline keeps its last known-good map
    assert mappings_path.read_text(encoding="utf-8") == good_content
    # and no reload was requested
    assert reload_signal.consume_config_reload() == ""


def test_put_mapping_out_of_bounds_rejected(temp_camera_env):
    bad = dict(VALID_PAYLOAD)
    bad["roi"] = [[0.1, 0.1], [1.4, 0.1], [0.9, 0.8], [0.1, 0.8]]
    resp = client.put("/api/dashboard/mapping", json=bad)
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is False
    assert any("outside" in e for e in data["errors"])


def test_put_mapping_accepts_list_form_of_lines(temp_camera_env):
    """The editor sends the list form; both must pass validation."""
    payload = dict(VALID_PAYLOAD)
    payload["lines"] = [
        {"id": "in", "points": [[0.875, 0.777778], [0.125, 0.777778]], "direction": "IN"},
        {"id": "out", "points": [[0.125, 0.777778], [0.875, 0.777778]], "direction": "OUT"},
    ]
    resp = client.put("/api/dashboard/mapping", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is True
    persisted = _load_yaml(temp_camera_env.parent / "mappings.yaml")
    entry = persisted["cameras"]["CAMERA 01"]
    assert {l["id"] for l in entry["lines"]} == {"in", "out"}


def test_config_endpoint_carries_mapping_fields(temp_camera_env):
    client.put("/api/dashboard/mapping", json=VALID_PAYLOAD)
    reload_signal.reset_config_reload()
    resp = client.get("/api/dashboard/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "mapping_active" in data
    assert "reference_point" in data
    assert "source_shape_w" in data
    assert data["mapping_active"] is True
    assert data["reference_point"] == "bottom_center"


def _load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))
