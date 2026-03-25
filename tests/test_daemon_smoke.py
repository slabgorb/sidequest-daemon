"""Smoke test — verify the daemon starts and responds to ping."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time

import pytest


SOCKET_PATH = "/tmp/sidequest-renderer.sock"


@pytest.fixture()
def daemon_process(tmp_path):
    """Start the daemon in a subprocess, yield, then shut it down."""
    env = {
        **os.environ,
        "SIDEQUEST_GENRE_PACKS": str(tmp_path),
    }
    proc = subprocess.Popen(
        ["sidequest-renderer", "--output-dir", str(tmp_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for socket to appear
    for _ in range(20):
        if os.path.exists(SOCKET_PATH):
            break
        time.sleep(0.25)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_daemon_ping(daemon_process):
    """Send a ping over the Unix socket and verify the response."""
    reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
    try:
        request = json.dumps({"id": "smoke", "method": "ping", "params": {}})
        writer.write((request + "\n").encode())
        await writer.drain()

        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        response = json.loads(line)
        assert response["id"] == "smoke"
        assert response["result"]["status"] == "ok"
    finally:
        writer.close()
        await writer.wait_closed()
