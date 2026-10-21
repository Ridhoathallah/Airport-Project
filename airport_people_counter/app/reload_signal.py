"""In-process signal for configuration reloads driven by the API.

``run_api_server`` runs in a *thread* of the same interpreter as the
detection loop, so a ``threading.Event`` is all the plumbing needed for
``PUT /api/dashboard/mapping`` to tell the loop "re-read the config".
The loop polls ``consume_config_reload()`` once per frame — the same
pattern ``shutdown.request_shutdown`` uses for process stops.

The signal is intentionally process-local: nothing is persisted here,
and a restart of the pipeline reloads from disk by construction.
"""

from __future__ import annotations

import threading

_event = threading.Event()
_source = "api"
_lock = threading.Lock()


def request_config_reload(source: str = "api") -> None:
    """Ask the detection loop to reload its camera config + mapping."""
    global _source
    with _lock:
        _source = source
    _event.set()


def consume_config_reload() -> str:
    """Return the reload source when a reload is pending, else ``""``.

    Consuming clears the flag, so exactly one loop iteration reacts to
    each request even if the loop is slower than the API thread.
    """
    if not _event.is_set():
        return ""
    with _lock:
        source = _source
    _event.clear()
    return source


def reset_config_reload() -> None:
    """Clear any pending request (tests, startup)."""
    global _source
    with _lock:
        _source = "api"
    _event.clear()
