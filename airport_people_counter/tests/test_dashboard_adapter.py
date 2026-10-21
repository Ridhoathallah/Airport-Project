"""
Unit tests for dashboard_adapter.py
"""

import sys
import os
import pytest

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from dashboard_adapter import (
    build_dashboard_snapshot,
    reset_adapter_state,
    _map_cv_state_to_ui,
    _cv_state_is_unconfigured,
    _compute_trend,
    _reason_for,
    _derive_system_status,
)
from dashboard_models import (
    DashboardSnapshot,
    QueueTelemetry,
    SystemTelemetry,
    CameraTelemetry,
    RecordingTelemetry,
    Sample,
    EventInfo,
    TrackInfo,
)


class MockCameraState:
    def __init__(self, link="ONLINE", fps=24.0, resolution="1920x1080"):
        self.link = link
        self.fps = fps
        self.resolution = resolution


class MockTrack:
    def __init__(self, id=1, x=0.5, y=0.5, w=0.1, h=0.2, in_roi=True):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.in_roi = in_roi
        self.confidence = 0.95
        self.age = 10
        self.hits = 8
        self.state = "active"


class MockEvent:
    def __init__(self, id=1, t=1700000000000, kind="entry", severity="info", label="Test", detail=None):
        self.id = id
        self.t = t
        self.kind = kind
        self.severity = severity
        self.label = label
        self.detail = detail


class MockQueueSnapshot:
    def __init__(
        self,
        people_in_queue=5,
        queue_state="NORMAL",
        occupancy_percent=16.7,
        congestion_duration_s=0.0,
        entry_rate=2.5,
        exit_rate=1.8,
        avg_wait_s=72.0,
        longest_wait_s=150.0,
        camera_state=None,
        fps=24.0,
        inference_ms=25.5,
        detection_count=5,
        tracked_people=4,
        timestamp="2024-01-15T10:30:00Z",
        tracks=None,
        events=None,
        staleness_ms=48,
    ):
        self.people_in_queue = people_in_queue
        self.queue_state = queue_state
        self.occupancy_percent = occupancy_percent
        self.congestion_duration_s = congestion_duration_s
        self.entry_rate = entry_rate
        self.exit_rate = exit_rate
        self.avg_wait_s = avg_wait_s
        self.longest_wait_s = longest_wait_s
        self.camera_state = camera_state or MockCameraState()
        self.fps = fps
        self.inference_ms = inference_ms
        self.detection_count = detection_count
        self.tracked_people = tracked_people
        self.timestamp = timestamp
        self.tracks = tracks or [MockTrack()]
        self.events = events or [MockEvent()]
        self.staleness_ms = staleness_ms


class MockCameraConfig:
    def __init__(self, queue_capacity=30):
        self.queue_capacity = queue_capacity


def test_map_cv_state_to_ui():
    """Test CV state to UI state mapping."""
    assert _map_cv_state_to_ui("EMPTY") == "NORMAL"
    assert _map_cv_state_to_ui("SPARSE") == "NORMAL"
    assert _map_cv_state_to_ui("NORMAL") == "NORMAL"
    assert _map_cv_state_to_ui("CROWDED") == "BUSY"
    assert _map_cv_state_to_ui("CONGESTED") == "CONGESTED"
    assert _map_cv_state_to_ui("BACKLOG") == "BACKLOG"
    assert _map_cv_state_to_ui("UNCONFIGURED") == "NORMAL"
    assert _map_cv_state_to_ui("DATA_UNAVAILABLE") == "NORMAL"
    assert _map_cv_state_to_ui("UNKNOWN") == "NORMAL"


def test_cv_state_is_unconfigured():
    """Test unconfigured state detection."""
    assert _cv_state_is_unconfigured("UNCONFIGURED") is True
    assert _cv_state_is_unconfigured("DATA_UNAVAILABLE") is True
    assert _cv_state_is_unconfigured("NORMAL") is False
    assert _cv_state_is_unconfigured("CONGESTED") is False


def test_compute_trend():
    """Test trend computation from net change."""
    assert _compute_trend(2.0) == "GROWING"
    assert _compute_trend(-2.0) == "CLEARING"
    assert _compute_trend(0.5) == "STEADY"
    assert _compute_trend(-0.5) == "STEADY"
    assert _compute_trend(1.0) == "STEADY"
    assert _compute_trend(-1.0) == "STEADY"


def test_reason_for():
    """Test reason text generation."""
    assert "exceeds capacity" in _reason_for("BACKLOG", 1.0, 95.0)
    assert "above 80%" in _reason_for("CONGESTED", 2.0, 85.0)
    assert "throughput below demand" in _reason_for("CONGESTED", -0.5, 85.0)
    assert "Elevated demand" in _reason_for("BUSY", 2.0, 60.0)
    assert "balanced" in _reason_for("BUSY", 0.0, 60.0)
    assert "clearing normally" in _reason_for("NORMAL", -2.0, 20.0)
    assert "Nominal" in _reason_for("NORMAL", 0.0, 20.0)


def test_derive_system_status():
    """Test system status derivation from FPS."""
    assert _derive_system_status(24.0) == "OK"
    assert _derive_system_status(10.0) == "OK"
    assert _derive_system_status(5.0) == "DEGRADED"
    assert _derive_system_status(0.1) == "DEGRADED"
    assert _derive_system_status(0.0) == "STALLED"


def test_build_dashboard_snapshot_normal():
    """Test building dashboard snapshot in NORMAL state."""
    reset_adapter_state()
    snap = MockQueueSnapshot(entry_rate=3.0, exit_rate=1.0)  # net_change = 2.0 > 1.0 => GROWING
    config = MockCameraConfig()
    history = [
        Sample(t=1700000000000, queue=5, occupancy=16.7, entry_rate=3.0, exit_rate=1.0, avg_wait=1.2)
    ]
    server_start_ts = 1700000000.0

    result = build_dashboard_snapshot(snap, config, history, server_start_ts)

    assert isinstance(result, DashboardSnapshot)
    assert result.scenario == "NORMAL"
    assert result.queue.queue == 5
    assert result.queue.capacity == 30
    assert result.queue.occupancy == 16.7
    assert result.queue.trend == "GROWING"
    assert result.queue.state == "NORMAL"
    assert result.queue.state_active is True
    assert result.system.detection_status == "OK"
    assert result.camera.link == "ONLINE"
    assert result.last_known_queue == 5
    assert len(result.history) == 1
    assert len(result.events) == 1
    assert len(result.tracks) == 1


def test_build_dashboard_snapshot_offline_freezes_values():
    """Test OFFLINE freezes last known queue and state."""
    reset_adapter_state()
    # First, establish a known state with ONLINE
    snap_online = MockQueueSnapshot(people_in_queue=10, queue_state="CONGESTED")
    config = MockCameraConfig()
    history = []
    server_start_ts = 1700000000.0

    result1 = build_dashboard_snapshot(snap_online, config, history, server_start_ts)
    assert result1.queue.queue == 10
    assert result1.queue.state == "CONGESTED"

    # Then go OFFLINE - should freeze at 10
    snap_offline = MockQueueSnapshot(
        people_in_queue=0,  # Different value
        queue_state="DATA_UNAVAILABLE",
        camera_state=MockCameraState(link="OFFLINE"),
    )
    result2 = build_dashboard_snapshot(snap_offline, config, history, server_start_ts)
    assert result2.queue.queue == 10  # Frozen at last known
    assert result2.queue.state == "CONGESTED"  # Frozen at last known
    assert result2.scenario == "OFFLINE"
    assert result2.queue.state_active is False
    assert "Camera signal lost" in result2.queue.reason


def test_build_dashboard_snapshot_degraded():
    """Test DEGRADED link state."""
    reset_adapter_state()
    snap = MockQueueSnapshot(
        queue_state="CONGESTED",
        camera_state=MockCameraState(link="DEGRADED"),
        fps=5.0,
    )
    config = MockCameraConfig()
    history = []
    server_start_ts = 1700000000.0

    result = build_dashboard_snapshot(snap, config, history, server_start_ts)

    assert result.scenario == "DEGRADED"
    assert result.camera.link == "DEGRADED"
    assert result.system.detection_status == "DEGRADED"
    assert "low-confidence" in result.queue.reason


def test_build_dashboard_snapshot_stalled():
    """Test STALLED system status when FPS is 0."""
    reset_adapter_state()
    snap = MockQueueSnapshot(fps=0.0)
    config = MockCameraConfig()
    history = []
    server_start_ts = 1700000000.0

    result = build_dashboard_snapshot(snap, config, history, server_start_ts)

    assert result.system.detection_status == "STALLED"
    assert result.system.tracking_status == "STALLED"
    assert result.system.analytics_status == "STALLED"


def test_build_dashboard_snapshot_unconfigured_source_state():
    """Test source_state is set for unconfigured states."""
    reset_adapter_state()
    snap = MockQueueSnapshot(queue_state="UNCONFIGURED")
    config = MockCameraConfig()
    history = []
    server_start_ts = 1700000000.0

    result = build_dashboard_snapshot(snap, config, history, server_start_ts)

    assert result.queue.source_state == "UNCONFIGURED"
    assert result.queue.state == "NORMAL"  # UI state is NORMAL


def test_build_dashboard_snapshot_backlog():
    """Test BACKLOG scenario."""
    reset_adapter_state()
    snap = MockQueueSnapshot(
        queue_state="BACKLOG",
        people_in_queue=35,
        occupancy_percent=116.7,
        entry_rate=5.0,
        exit_rate=2.0,
    )
    config = MockCameraConfig(queue_capacity=30)
    history = []
    server_start_ts = 1700000000.0

    result = build_dashboard_snapshot(snap, config, history, server_start_ts)

    assert result.scenario == "BACKLOG"
    assert result.queue.state == "BACKLOG"
    assert "exceeds capacity" in result.queue.reason


def test_reset_adapter_state():
    """Test adapter state reset."""
    reset_adapter_state()
    snap = MockQueueSnapshot(people_in_queue=10, queue_state="CONGESTED")
    config = MockCameraConfig()
    history = []
    server_start_ts = 1700000000.0

    build_dashboard_snapshot(snap, config, history, server_start_ts)
    reset_adapter_state()

    # After reset, OFFLINE should show 0
    snap_offline = MockQueueSnapshot(
        people_in_queue=0,
        queue_state="DATA_UNAVAILABLE",
        camera_state=MockCameraState(link="OFFLINE"),
    )
    result = build_dashboard_snapshot(snap_offline, config, history, server_start_ts)
    assert result.queue.queue == 0
    assert result.queue.state == "NORMAL"


def test_serialization():
    """Test that the result serializes to valid JSON."""
    import json
    reset_adapter_state()
    snap = MockQueueSnapshot()
    config = MockCameraConfig()
    history = []
    server_start_ts = 1700000000.0

    result = build_dashboard_snapshot(snap, config, history, server_start_ts)
    json_str = result.model_dump_json()

    # Verify key fields are in JSON
    assert "scenario" in json_str
    assert "queue" in json_str
    assert "system" in json_str
    assert "camera" in json_str
    assert "recording" in json_str
    assert "last_known_queue" in json_str
    assert "last_update" in json_str
    assert "history" in json_str
    assert "events" in json_str
    assert "tracks" in json_str
    assert "frame_ts" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])