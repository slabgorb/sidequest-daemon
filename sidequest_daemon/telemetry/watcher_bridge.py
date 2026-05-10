"""Cross-process watcher event bridge — daemon → server.

The watcher hub lives in the server process (sidequest.telemetry.watcher_hub).
The daemon cannot import it, so this module POSTs each event to a server
HTTP endpoint that forwards to publish_event().

Per CLAUDE.md no-silent-fallbacks: network errors are LOGGED LOUDLY, not
swallowed. We never crash the calling path on telemetry failure (telemetry
must not break renders), but we make the failure visible.

Uses urllib.request (stdlib) rather than requests — neither daemon nor
server pulls requests as a runtime dep, and a fire-and-forget telemetry
POST does not justify adding one. urllib.request is sync and crude; that
is acceptable for a 2-second timeout fire-and-forget call from an error path.
(Deviation from plan's requests.post reference — documented here per plan
instructions.)
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_SERVER_URL = "http://127.0.0.1:8765"
_TIMEOUT_SECONDS = 2.0


def _server_base_url() -> str:
    return os.environ.get("SIDEQUEST_SERVER_URL", _DEFAULT_SERVER_URL).rstrip("/")


def _post(url: str, body: dict[str, Any]) -> None:
    """Sync POST with a short timeout. Raises on any network error."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS):
        pass


def emit_watcher_event(
    event_type: str, fields: dict[str, Any], *, component: str = "daemon"
) -> None:
    """Forward a watcher event to the server's hub via HTTP.

    Failures are logged at WARNING — never raised. Telemetry must not
    break the calling path, but the failure must be visible (this is
    the fix for the silent ImportError pattern flagged in the wiring audit).
    """
    url = f"{_server_base_url()}/internal/watcher/emit"
    body = {"event_type": event_type, "fields": fields, "component": component}
    try:
        _post(url, body)
    except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as exc:
        log.warning("watcher_bridge POST failed (%s): %s", type(exc).__name__, exc)
