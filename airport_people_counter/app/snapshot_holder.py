"""
Thread-safe holder for the latest QueueSnapshot and annotated frame.

Used by camera engine (writer) and FastAPI server (reader).
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

from snapshot import QueueSnapshot


class SnapshotHolder:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot: Optional[QueueSnapshot] = None
        self._frame: Optional[np.ndarray] = None

    def update(self, snapshot: QueueSnapshot, frame: Optional[np.ndarray] = None) -> None:
        with self._lock:
            self._snapshot = snapshot
            if frame is not None:
                self._frame = frame.copy()

    def get_latest_snapshot(self) -> Optional[QueueSnapshot]:
        with self._lock:
            return self._snapshot

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_both(self) -> Tuple[Optional[QueueSnapshot], Optional[np.ndarray]]:
        with self._lock:
            snap = self._snapshot
            frame = self._frame.copy() if self._frame is not None else None
            return snap, frame


# Global instance
_holder = SnapshotHolder()


def get_holder() -> SnapshotHolder:
    return _holder
