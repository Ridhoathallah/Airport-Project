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

from snapshot_buffer import HistoryBuffer, get_history_buffer, reset_history_buffer
from dashboard_models import Sample


def test_buffer_capacity():
    """Buffer should hold exactly max_len samples."""
    buf = HistoryBuffer(max_len=10)
    for i in range(15):
        buf.append(Sample(t=i, queue=i, occupancy=0.0, entry_rate=0.0, exit_rate=0.0, avg_wait=0.0))
    assert len(buf) == 10
    snap = buf.snapshot()
    assert len(snap) == 10
    # Oldest 5 should be evicted
    assert snap[0].t == 5
    assert snap[-1].t == 14


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
        buf.append(Sample(t=i, queue=i, occupancy=0.0, entry_rate=0.0, exit_rate=0.0, avg_wait=0.0))
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])