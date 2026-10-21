"""
Thread-safe ring buffer for dashboard snapshot history.

Samples are recorded on a fixed 1 Hz grid, so the buffer covers a
predictable span of wall-clock time: 3600 samples is 60 minutes. A buffer
that filled at the camera frame rate would cover seconds, and the
dashboard's 5m / 15m / 30m / 60m ranges would describe frames rather than
time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

from dashboard_models import Sample

#: One sample per second.
SAMPLE_INTERVAL_MS = 1000


@dataclass
class HistoryBuffer:
    """Thread-safe fixed-capacity ring buffer of Sample objects."""

    max_len: int = 3600
    min_interval_ms: int = SAMPLE_INTERVAL_MS
    _buffer: List[Sample] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, sample: Sample) -> bool:
        """Add a new sample, evicting the oldest if at capacity.

        A sample that arrives less than ``min_interval_ms`` after the stored
        one is dropped, which keeps the buffer on a fixed time grid no matter
        what the frame rate does. The stored timestamps are the sample's own,
        so the grid stays correct across a stall or a frame-rate change.

        Returns True when the sample was stored.
        """
        with self._lock:
            if self._buffer:
                elapsed = sample.t - self._buffer[-1].t
                if elapsed < self.min_interval_ms:
                    return False
            self._buffer.append(sample)
            if len(self._buffer) > self.max_len:
                self._buffer.pop(0)
            return True

    def snapshot(self) -> List[Sample]:
        """Return a copy of the current buffer contents (oldest first)."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._buffer


def downsample(samples: List[Sample], max_points: int = 240) -> List[Sample]:
    """Reduce a history to at most ``max_points`` evenly spaced samples.

    The first and the newest sample are always kept, so the series still
    describes the full span it covers. Used to keep the polled snapshot
    payload small while the buffer itself stays at full resolution.
    """
    if max_points < 2:
        return list(samples[-max_points:]) if samples else []
    if len(samples) <= max_points:
        return list(samples)

    step = (len(samples) - 1) / (max_points - 1)
    out: List[Sample] = []
    seen = set()
    for i in range(max_points):
        idx = int(round(i * step))
        if idx not in seen:
            seen.add(idx)
            out.append(samples[idx])
    return out


# Global instance
_history_buffer: Optional[HistoryBuffer] = None


def get_history_buffer() -> HistoryBuffer:
    """Get the global history buffer instance."""
    global _history_buffer
    if _history_buffer is None:
        _history_buffer = HistoryBuffer()
    return _history_buffer


def reset_history_buffer() -> None:
    """Reset the global history buffer (for testing)."""
    global _history_buffer
    _history_buffer = None
