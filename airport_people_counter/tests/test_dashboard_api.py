"""
Unit tests for dashboard API endpoints.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from api_server import app
from dashboard_models import DashboardSnapshot, ConfigResponse, HealthResponse


client = TestClient(app)


def _make_mock_snap():
    """Create a mock snapshot with all required fields."""
    mock_snap = MagicMock()
    mock_snap.people_in_queue = 5
    mock_snap.queue_state = "NORMAL"
    mock_snap.occupancy_percent = 16.7
    mock_snap.congestion_duration_s = 0.0
    mock_snap.entry_rate = 3.0
    mock_snap.exit_rate = 1.0
    mock_snap.avg_wait_s = 72.0
    mock_snap.longest_wait_s = 150.0
    mock_snap.fps = 24.0
    mock_snap.inference_ms = 25.5
    mock_snap.tracking_ms = 5.0
    mock_snap.analytics_ms = 2.0
    mock_snap.detection_count = 5
    mock_snap.tracked_people = 4
    mock_snap.timestamp = "2024-01-15T10:30:00Z"
    mock_snap.tracks = [MagicMock(id=1, x=0.5, y=0.5, w=0.1, h=0.2, in_roi=True, confidence=0.95, age=10, hits=8, state="active")]
    mock_snap.events = [MagicMock(id=1, t=1700000000000, kind="entry", severity="info", label="Test", detail=None)]
    mock_snap.staleness_ms = 48
    mock_snap.camera_state = MagicMock(link="ONLINE", fps=24.0, resolution="1920x1080", frame_age_ms=48)
    mock_snap.config_status = MagicMock(roi_configured=True, line_configured=True, calibrated=True, classifier_enabled=False)
    mock_snap.calibration = "calibrated"  # Pydantic expects string
    mock_snap.frame_id = 1
    mock_snap.freshness = "LIVE"
    mock_snap.total_in = 100
    mock_snap.total_out = 95
    mock_snap.spacing_px = 1.5
    mock_snap.movement = "normal"  # Pydantic expects string
    return mock_snap


def _make_mock_config():
    """Create a mock camera config."""
    mock_config = MagicMock()
    mock_config.camera_id = "CAMERA 01"
    mock_config.queue_capacity = 30
    mock_config.width = 1920
    mock_config.height = 1080
    mock_config.roi_configured = True
    mock_config.line_configured = True
    mock_config.roi = [[100, 100], [500, 100], [500, 400], [100, 400]]
    mock_config.counting_line = [[100, 250], [500, 250]]
    mock_config.xray_side_point = [1200, 500]
    mock_config.occupancy_sparse = 10
    mock_config.occupancy_normal = 30
    mock_config.occupancy_crowded = 60
    mock_config.occupancy_congested = 80
    mock_config.hysteresis = 5
    mock_config.debounce_frames = 3
    mock_config.backlog_min_people = 25
    mock_config.backlog_min_occupancy = 85
    mock_config.backlog_min_wait_seconds = 300
    mock_config.backlog_persistence_seconds = 60
    return mock_config


def test_dashboard_snapshot_endpoint():
    """Test /api/dashboard/snapshot returns DashboardSnapshot shape."""
    with patch('api_server.get_holder') as mock_holder:
        mock_snap = _make_mock_snap()
        mock_holder.return_value.get_latest_snapshot.return_value = mock_snap

        with patch('queue_prototype.load_camera_config') as mock_load:
            mock_load.return_value = _make_mock_config()

            response = client.get("/api/dashboard/snapshot")
            assert response.status_code == 200
            data = response.json()
            assert "scenario" in data
            assert "queue" in data
            assert "system" in data
            assert "camera" in data
            assert "recording" in data
            assert "last_known_queue" in data
            assert "last_update" in data
            assert "history" in data
            assert "events" in data
            assert "tracks" in data
            assert "frame_ts" in data


def test_dashboard_config_endpoint():
    """Test /api/dashboard/config returns ConfigResponse shape."""
    with patch('queue_prototype.load_camera_config') as mock_load:
        mock_load.return_value = _make_mock_config()

        response = client.get("/api/dashboard/config")
        assert response.status_code == 200
        data = response.json()
        assert data["camera_id"] == "CAMERA 01"
        assert data["roi_configured"] is True
        assert data["line_configured"] is True
        assert data["width"] == 1920
        assert data["height"] == 1080
        assert data["xray_side"] == "right"


def test_dashboard_health_endpoint():
    """Test /api/dashboard/health returns HealthResponse shape."""
    with patch('api_server.get_holder') as mock_holder:
        mock_snap = MagicMock()
        mock_snap.camera_state = MagicMock(link="ONLINE", fps=24.0)
        mock_snap.staleness_ms = 48
        mock_snap.fps = 24.0
        mock_snap.freshness = "LIVE"
        mock_holder.return_value.get_latest_snapshot.return_value = mock_snap

        response = client.get("/api/dashboard/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert data["camera"] is True
        assert data["api"] is True
        assert data["last_frame_age_ms"] == 48


def test_dashboard_health_offline():
    """Test /api/dashboard/health when camera is offline."""
    with patch('api_server.get_holder') as mock_holder:
        mock_snap = MagicMock()
        mock_snap.camera_state = MagicMock(link="OFFLINE", fps=0.0)
        mock_snap.staleness_ms = 5000
        mock_snap.fps = 0.0
        mock_snap.freshness = "OFFLINE"
        mock_holder.return_value.get_latest_snapshot.return_value = mock_snap

        response = client.get("/api/dashboard/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "DEGRADED"
        assert data["camera"] is False
        assert data["api"] is True
        assert data["last_frame_age_ms"] == 5000


def test_dashboard_history_endpoint():
    """Test /api/dashboard/history returns samples."""
    with patch('api_server.get_history_buffer') as mock_buffer:
        mock_buf = MagicMock()
        mock_buf.snapshot.return_value = [
            MagicMock(model_dump=lambda: {"t": 1700000000000, "queue": 5, "occupancy": 16.7, "entry_rate": 2.5, "exit_rate": 1.8, "avg_wait": 1.2}),
        ]
        mock_buffer.return_value = mock_buf

        response = client.get("/api/dashboard/history")
        assert response.status_code == 200
        data = response.json()
        assert "samples" in data
        assert len(data["samples"]) == 1
        assert data["samples"][0]["queue"] == 5


def test_existing_snapshot_endpoint_still_works():
    """Test that the original /api/snapshot endpoint still works."""
    with patch('api_server.get_holder') as mock_holder:
        mock_snap = _make_mock_snap()
        mock_snap.entry_rate = 2.5
        mock_snap.exit_rate = 1.8
        mock_holder.return_value.get_latest_snapshot.return_value = mock_snap

        response = client.get("/api/snapshot")
        assert response.status_code == 200
        data = response.json()
        assert data["people_in_queue"] == 5
        assert data["queue_state"] == "NORMAL"
        assert data["freshness"] == "LIVE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])