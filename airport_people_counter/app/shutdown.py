"""
Cooperative shutdown coordination between the API server and the detection loop.

The dashboard's "Shut Down" button calls ``POST /api/shutdown``. The API
server and the detection loop live in the *same* process (the loop owns
the camera and the model; the server is a daemon thread), so the request
must stop the loop rather than the process.

Historically the endpoint called ``os._exit(0)``. That skipped every
cleanup step: the camera capture was never released, buffered writes were
lost, and the process died mid-frame. This module replaces that with a
cooperative flag:

* the detection loop polls :func:`shutdown_requested` once per frame and
  exits its loop normally, releasing the camera and closing its outputs;
* the API server asks uvicorn to stop serving, which lets in-flight
  requests finish;
* a watchdog thread remains as a last-resort backstop so a wedged process
  still goes away, and it says so in the log.

The flag is process-global, which is correct here: there is exactly one
detection loop and one API server per process.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

#: Set when a shutdown has been requested. ``threading.Event`` is used
#: because it is the only primitive that is both a flag and a waitable.
_shutdown_event = threading.Event()

#: Why the shutdown was requested (for the log).
_shutdown_reason = ""

#: Guards ``_shutdown_reason``.
_reason_lock = threading.Lock()


def request_shutdown(reason: str = "requested") -> None:
    """Ask the process to shut down. Safe to call more than once."""
    global _shutdown_reason
    with _reason_lock:
        if not _shutdown_event.is_set():
            _shutdown_reason = reason
    _shutdown_event.set()


def shutdown_requested() -> bool:
    """True once a shutdown has been requested."""
    return _shutdown_event.is_set()


def shutdown_reason() -> str:
    """The reason given for the shutdown request (empty if none)."""
    with _reason_lock:
        return _shutdown_reason


def wait_for_shutdown(timeout: Optional[float] = None) -> bool:
    """Block until a shutdown is requested. Returns True if it was."""
    return _shutdown_event.wait(timeout)


def reset_shutdown() -> None:
    """Clear the flag (used by tests)."""
    global _shutdown_reason
    with _reason_lock:
        _shutdown_reason = ""
    _shutdown_event.clear()


def start_watchdog(timeout_s: float = 10.0, exit_func=None) -> threading.Thread:
    """Force the process down if the cooperative shutdown does not finish.

    This is the safety net that ``os._exit(0)`` used to be -- but it only
    fires when the graceful path has genuinely failed, and it reports that
    it did so.
    """
    import os
    import sys

    exit_fn = exit_func or (lambda code: os._exit(code))

    def _watchdog():
        if not _shutdown_event.wait(timeout_s):
            print(
                f"[WARN] graceful shutdown did not finish within {timeout_s:.0f}s "
                "-- forcing exit",
                file=sys.stderr,
                flush=True,
            )
            exit_fn(0)

    thread = threading.Thread(target=_watchdog, name="shutdown-watchdog", daemon=True)
    thread.start()
    return thread
