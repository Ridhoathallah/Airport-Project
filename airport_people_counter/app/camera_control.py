"""In-process signal for camera source/rotation controls driven by the API.

Mirrors ``reload_signal``: ``run_api_server`` runs in a thread of the
same interpreter as the detection loop, so a ``threading.Event`` plus a
payload is all the plumbing needed for the camera-control endpoints to
tell the loop "swap the source" or "rotate the frames".

Process-local by design: nothing is persisted here and a pipeline
restart reloads everything from disk by construction.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

_lock = threading.Lock()
_event = threading.Event()
_payload: Dict = {}
_source = "api"


def request_camera_control(payload: Dict, source: str = "api") -> None:
    """Ask the detection loop to perform a camera control action.

    Recognised payloads carry ``{"action": "switch", "source": "1"}`` or
    ``{"action": "rotate", "rotation": 90}``. Exactly one payload is
    pending at a time; a newer request replaces a pending one.
    """
    global _payload, _source
    with _lock:
        _payload = dict(payload)
        _source = source
    _event.set()


def consume_camera_control() -> Tuple[str, Dict, str]:
    """Pop the pending action: ``(action, payload, source)`` when one is
    pending, else ``("", dict(), "")``."""
    global _payload, _source
    if not _event.is_set():
        return "", {}, ""
    with _lock:
        action = str(_payload.get("action", ""))
        payload = dict(_payload)
        src = _source
        _payload = {}
        _source = "api"
    _event.clear()
    return action, payload, src
