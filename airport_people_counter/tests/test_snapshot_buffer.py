"""
Unit tests for snapshot_buffer.py
"""

import sys
import os
import threading
import time
import pytest

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from snapshot_buffer import HistoryBuffer, get_history_buffer, reset_history_buffer, downsample
from dashboard_models import Sample


def test_buffer_capacity():
    """Buffer should hold exactly max_len samples."""
    buf = HistoryBuffer(max_len=10)
    # Samples are spaced by the 1 Hz grid the buffer records on.
    for i in range(15):
        buf.append(Sample(t=i * 1000, queue=i, occupancy=0.0, entry_rate=0.0,
                          exit_rate=0.0, avg_wait=0.0))
    assert len(buf) == 10
    snap = buf.snapshot()
    assert len(snap) == 10
    # Oldest 5 should be evicted
    assert snap[0].t == 5000
    assert snap[-1].t == 14000


def test_buffer_drops_samples_inside_the_interval():
    """A sample closer than the grid interval is not stored.

    The buffer is on a fixed 1 Hz grid so that the dashboard's 5m / 15m /
    30m / 60m ranges describe wall-clock time rather than frames.
    """
    buf = HistoryBuffer(max_len=10)
    assert buf.append(Sample(t=0, queue=0, occupancy=0.0, entry_rate=0.0,
                             exit_rate=0.0, avg_wait=0.0)) is True
    # 500 ms later: inside the 1 s grid, so it is dropped.
    assert buf.append(Sample(t=500, queue=1, occupancy=0.0, entry_rate=0.0,
                             exit_rate=0.0, avg_wait=0.0)) is False
    assert len(buf) == 1
    # 1000 ms later: on the grid, so it is stored.
    assert buf.append(Sample(t=1000, queue=1, occupancy=0.0, entry_rate=0.0,
                             exit_rate=0.0, avg_wait=0.0)) is True
    assert len(buf) == 2


def test_buffer_default_capacity_covers_one_hour():
    """The default buffer is one hour of 1 Hz samples."""
    buf = HistoryBuffer()
    assert buf.max_len == 3600
    assert buf.min_interval_ms == 1000


def test_buffer_thread_safety():
    """Concurrent writers and reader should not corrupt the buffer."""
    buf = HistoryBuffer(max_len=1000)
    errors = []

    def writer(start: int, count: int):
        try:
            for i in range(count):
                buf.append(Sample(t=start + i, queue=i, occupancy=0.0, entry_rate=0.0, exit_rate=0.0, avg_wait=0.0))
                time.sleep(0.0001)
        except Exception as e:
            errors.append(e)

    def reader():
        for _ in range(100):
            snap = buf.snapshot()
            # Just verify we can read without exception
            _ = len(snap)
            time.sleep(0.001)

    threads = [
        threading.Thread(target=writer, args=(0, 500)),
        threading.Thread(target=writer, args=(500, 500)),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(buf) <= 1000


def test_buffer_snapshot_is_copy():
    """snapshot() should return a copy, not the internal list."""
    buf = HistoryBuffer(max_len=10)
    buf.append(Sample(t=1, queue=1, occupancy=0.0, entry_rate=0.0, exit_rate=0.0, avg_wait=0.0))
    snap = buf.snapshot()
    snap.append(Sample(t=2, queue=2, occupancy=0.0, entry_rate=0.0, exit_rate=0.0, avg_wait=0.0))
    # Original buffer should be unchanged
    assert len(buf) == 1


def test_buffer_clear():
    buf = HistoryBuffer(max_len=10)
    for i in range(5):
        buf.append(Sample(t=i * 1000, queue=i, occupancy=0.0, entry_rate=0.0,
                          exit_rate=0.0, avg_wait=0.0))
    assert len(buf) == 5
    buf.clear()
    assert len(buf) == 0
    assert buf.is_empty()


def test_global_instance():
    reset_history_buffer()
    buf1 = get_history_buffer()
    buf2 = get_history_buffer()
    assert buf1 is buf2
    reset_history_buffer()


def _sample(t_ms, queue=0):
    return Sample(t=t_ms, queue=queue, occupancy=0.0, entry_rate=0.0,
                  exit_rate=0.0, avg_wait=0.0)


def test_downsample_keeps_every_sample_when_short():
    samples = [_sample(i * 1000, i) for i in range(10)]
    out = downsample(samples, max_points=240)
    assert out == samples


def test_downsample_keeps_the_first_and_the_newest():
    samples = [_sample(i * 1000, i) for i in range(1000)]
    out = downsample(samples, max_points=240)
    assert len(out) == 240
    assert out[0].t == samples[0].t
    assert out[-1].t == samples[-1].t


def test_downsample_preserves_time_order_and_spacing():
    samples = [_sample(i * 1000, i) for i in range(1000)]
    out = downsample(samples, max_points=240)
    times = [s.t for s in out]
    assert times == sorted(times)
    gaps = {times[i + 1] - times[i] for i in range(len(times) - 1)}
    # The series covers the whole span evenly. Rounding to 240 points from
    # 1000 means the step is not a whole number of samples, so consecutive
    # gaps differ by at most one sample interval.
    assert max(gaps) - min(gaps) <= 1000
    assert min(gaps) > 0


def test_downsample_of_an_empty_buffer():
    assert downsample([], max_points=240) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])