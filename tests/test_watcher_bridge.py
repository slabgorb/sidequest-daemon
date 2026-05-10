import urllib.error
from unittest.mock import patch

from sidequest_daemon.telemetry import watcher_bridge
from sidequest_daemon.telemetry.watcher_bridge import emit_watcher_event


def test_emit_watcher_event_posts_to_server():
    with patch("sidequest_daemon.telemetry.watcher_bridge._post") as mock_post:
        emit_watcher_event("test.event", {"k": "v"})
        mock_post.assert_called_once()
        url, body = mock_post.call_args[0]
        assert url.endswith("/internal/watcher/emit")
        assert body == {"event_type": "test.event", "fields": {"k": "v"}, "component": "daemon"}


def test_emit_watcher_event_logs_loudly_on_network_error():
    """Daemon must never crash if the server is down. But the failure
    must log loudly — this is the inverse of the audit-flagged silent swallow."""
    with patch("sidequest_daemon.telemetry.watcher_bridge._post") as mock_post:
        mock_post.side_effect = ConnectionError("server down")
        with patch("sidequest_daemon.telemetry.watcher_bridge.log") as mock_log:
            emit_watcher_event("test.event", {})
            mock_log.warning.assert_called_once()
            assert "watcher_bridge" in mock_log.warning.call_args[0][0]


def test_emit_watcher_event_logs_loudly_on_urllib_error():
    """urllib.error.URLError is the actual failure mode when the server
    is not running (connection refused). ConnectionError covers a different
    branch of the except tuple — pin URLError explicitly."""
    with patch("sidequest_daemon.telemetry.watcher_bridge._post") as mock_post:
        mock_post.side_effect = urllib.error.URLError("connection refused")
        with patch("sidequest_daemon.telemetry.watcher_bridge.log") as mock_log:
            emit_watcher_event("test.event", {})
            mock_log.warning.assert_called_once()
            assert "watcher_bridge" in mock_log.warning.call_args[0][0]


def test_server_base_url_honors_env_override(monkeypatch):
    monkeypatch.setenv("SIDEQUEST_SERVER_URL", "http://other:9999/")
    assert watcher_bridge._server_base_url() == "http://other:9999"
