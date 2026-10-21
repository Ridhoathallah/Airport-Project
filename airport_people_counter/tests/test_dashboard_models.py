"""
Unit tests for dashboard_models.py
"""

import pytest
from app.dashboard_models import (
    DashboardSnapshot,
    QueueTelemetry,
    SystemTelemetry,
    CameraTelemetry,
    RecordingTelemetry,
    Sample,
    EventInfo,
    TrackInfo,
    ConfigResponse,
    HealthResponse,
)


def test_queue_telemetry_valid():
    q = QueueTelemetry(
        queue=5,
        capacity=30,
        occupancy=16.7,
        congestion_sec=0.0,
        entry_rate=2.5,
        exit_rate=1.8,
        net_change=0.7,
        trend="GROWING",
        avg_wait=1.2,
        longest_wait=2.5,
        throughput=108,
        state="NORMAL",
        state_active=True,
        reason="Nominal",
        source_state="NORMAL",
    )
    assert q.queue == 5
    assert q.capacity == 30
    assert q.trend == "GROWING"
    assert q.state == "NORMAL"
    assert q.state_active is True


def test_queue_telemetry_invalid_trend():
    with pytest.raises(ValueError):
        QueueTelemetry(
            queue=5,
            capacity=30,
            occupancy=16.7,
            congestion_sec=0.0,
            entry_rate=2.5,
            exit_rate=1.8,
            net_change=0.7,
            trend="INVALID",
            avg_wait=1.2,
            longest_wait=2.5,
            throughput=108,
            state="NORMAL",
            state_active=True,
            reason="Nominal",
        )


def test_system_telemetry_valid():
    s = SystemTelemetry(
        inference_ms=25.5,
        model="YOLOv5s",
        tracker="SORT",
        detections=5,
        active_tracks=4,
        detection_status="OK",
        tracking_status="OK",
        analytics_status="OK",
    )
    assert s.model == "YOLOv5s"
    assert s.detection_status == "OK"


def test_camera_telemetry_valid():
    c = CameraTelemetry(
        link="ONLINE",
        fps=24.0,
        resolution="1920x1080",
        latency_ms=62.0,
        packet_loss=0.1,
        last_frame_age_ms=48,
        uptime_sec=3600,
    )
    assert c.link == "ONLINE"
    assert c.fps == 24.0


def test_dashboard_snapshot_valid():
    snap = DashboardSnapshot(
        scenario="NORMAL",
        queue=QueueTelemetry(
            queue=5,
            capacity=30,
            occupancy=16.7,
            congestion_sec=0.0,
            entry_rate=2.5,
            exit_rate=1.8,
            net_change=0.7,
            trend="GROWING",
            avg_wait=1.2,
            longest_wait=2.5,
            throughput=108,
            state="NORMAL",
            state_active=True,
            reason="Nominal",
            source_state="NORMAL",
        ),
        system=SystemTelemetry(
            inference_ms=25.5,
            model="YOLOv5s",
            tracker="SORT",
            detections=5,
            active_tracks=4,
            detection_status="OK",
            tracking_status="OK",
            analytics_status="OK",
        ),
        camera=CameraTelemetry(
            link="ONLINE",
            fps=24.0,
            resolution="1920x1080",
            latency_ms=62.0,
            packet_loss=0.1,
            last_frame_age_ms=48,
            uptime_sec=3600,
        ),
        recording=RecordingTelemetry(active=False, since_sec=0, size_gb=0.0),
        last_known_queue=5,
        last_update=1700000000000,
        history=[
            Sample(t=1700000000000, queue=5, occupancy=16.7, entry_rate=2.5, exit_rate=1.8, avg_wait=1.2)
        ],
        events=[
            EventInfo(id=1, timestamp=1700000000000, kind="entry", severity="info", label="Passenger entered")
        ],
        tracks=[
            TrackInfo(id=1, x=0.5, y=0.5, w=0.1, h=0.2, in_roi=True)
        ],
        frame_ts=1700000000000,
    )
    assert snap.scenario == "NORMAL"
    assert snap.queue.queue == 5
    assert len(snap.history) == 1
    assert len(snap.events) == 1
    assert len(snap.tracks) == 1


def test_config_response_valid():
    c = ConfigResponse(
        camera_id="CAMERA 01",
        roi_configured=True,
        line_configured=True,
        roi_polygon=[[100, 100], [500, 100], [500, 400], [100, 400]],
        counting_line=[[100, 250], [500, 250]],
        xray_side="right",
        width=1920,
        height=1080,
    )
    assert c.camera_id == "CAMERA 01"
    assert c.roi_configured is True
    assert c.xray_side == "right"


def test_health_response_valid():
    h = HealthResponse(status="OK", camera=True, api=True, last_frame_age_ms=50)
    assert h.status == "OK"
    assert h.camera is True


def test_serialization():
    """Test that models serialize to the expected snake_case JSON."""
    snap = DashboardSnapshot(
        scenario="NORMAL",
        queue=QueueTelemetry(
            queue=5,
            capacity=30,
            occupancy=16.7,
            congestion_sec=0.0,
            entry_rate=2.5,
            exit_rate=1.8,
            net_change=0.7,
            trend="GROWING",
            avg_wait=1.2,
            longest_wait=2.5,
            throughput=108,
            state="NORMAL",
            state_active=True,
            reason="Nominal",
            source_state="NORMAL",
        ),
        system=SystemTelemetry(
            inference_ms=25.5,
            model="YOLOv5s",
            tracker="SORT",
            detections=5,
            active_tracks=4,
            detection_status="OK",
            tracking_status="OK",
            analytics_status="OK",
        ),
        camera=CameraTelemetry(
            link="ONLINE",
            fps=24.0,
            resolution="1920x1080",
            latency_ms=62.0,
            packet_loss=0.1,
            last_frame_age_ms=48,
            uptime_sec=3600,
        ),
        recording=RecordingTelemetry(active=False, since_sec=0, size_gb=0.0),
        last_known_queue=5,
        last_update=1700000000000,
        history=[],
        events=[],
        tracks=[],
        frame_ts=1700000000000,
    )

    json_str = snap.model_dump_json()
    assert "queue" in json_str
    assert "capacity" in json_str
    assert "state_active" in json_str
    assert "detection_status" in json_str
    assert "latency_ms" in json_str
    assert "last_update" in json_str
    assert "frame_ts" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])