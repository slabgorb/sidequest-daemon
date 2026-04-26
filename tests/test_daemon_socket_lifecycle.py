"""Regression tests for the socket-lifecycle race fixed 2026-04-26 [P1].

Bug summary: the daemon would log "Daemon listening on
/tmp/sidequest-renderer.sock", ``lsof`` would confirm the process held a
unix socket bound at that path, but ``ls`` showed no file on disk. The
inode had been unlinked. Server clients then logged
``render.skipped reason=daemon_unavailable`` because ``connect()`` failed.

Root cause: cleanup paths in ``daemon.py`` (the shutdown ``finally`` block
and the ``send_shutdown`` "stale socket" branch) called
``SOCKET_PATH.unlink()`` unconditionally. Any process running through
``_run_daemon`` that exited before binding — or any ``--shutdown`` invoked
while the listening daemon was still loading models — would unlink the
path that the live daemon had bound to. The kernel kept the bound socket
fd alive so ``lsof`` was happy, but new ``connect()`` calls failed because
the directory entry was gone.

Fix: a module-level ``_owns_socket`` flag set only after a successful
``start_unix_server`` bind, plus a ``_live_daemon_pid()`` liveness probe
on the PID file. Cleanup only proceeds when the current process owns the
bind, or when no other live daemon process is running.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from sidequest_daemon.media import daemon as daemon_mod


SOCKET_PATH = Path("/tmp/sidequest-renderer.sock")
PID_PATH = Path("/tmp/sidequest-renderer.pid")


def test_owns_socket_flag_starts_false():
    """The bind-ownership flag must start False — a fresh import must not
    grant cleanup rights to the importer."""
    # Re-import in a subprocess to get a clean module state.
    out = subprocess.check_output(
        [
            "python",
            "-c",
            "from sidequest_daemon.media import daemon; "
            "print(daemon._owns_socket)",
        ],
        text=True,
    ).strip()
    assert out == "False"


def test_live_daemon_pid_returns_none_when_no_pid_file(tmp_path, monkeypatch):
    """Liveness probe must return None when PID_PATH does not exist."""
    monkeypatch.setattr(daemon_mod, "PID_PATH", tmp_path / "no-such.pid")
    assert daemon_mod._live_daemon_pid() is None


def test_live_daemon_pid_returns_none_for_dead_pid(tmp_path, monkeypatch):
    """Liveness probe must return None when PID_PATH points to a dead PID."""
    pid_file = tmp_path / "dead.pid"
    # Spawn and reap a short-lived process to get a guaranteed-dead PID.
    proc = subprocess.Popen(["true"])
    proc.wait()
    pid_file.write_text(str(proc.pid))
    monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
    assert daemon_mod._live_daemon_pid() is None


def test_live_daemon_pid_returns_none_for_self(tmp_path, monkeypatch):
    """Liveness probe must return None if the PID file points at us — we
    are not 'another live daemon'."""
    pid_file = tmp_path / "self.pid"
    pid_file.write_text(str(os.getpid()))
    monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
    assert daemon_mod._live_daemon_pid() is None


def test_live_daemon_pid_detects_running_process(tmp_path, monkeypatch):
    """Liveness probe must return the PID when it points at a live process."""
    pid_file = tmp_path / "alive.pid"
    proc = subprocess.Popen(["sleep", "10"])
    try:
        pid_file.write_text(str(proc.pid))
        monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
        assert daemon_mod._live_daemon_pid() == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_live_daemon_pid_handles_garbage(tmp_path, monkeypatch):
    """Liveness probe must return None on unparseable PID file content."""
    pid_file = tmp_path / "garbage.pid"
    pid_file.write_text("not-a-number\n")
    monkeypatch.setattr(daemon_mod, "PID_PATH", pid_file)
    assert daemon_mod._live_daemon_pid() is None


@pytest.fixture()
def daemon_process(tmp_path):
    """Boot the real daemon in a subprocess (no warmup → fast). Tear it
    down after the test."""
    env = {
        **os.environ,
        "SIDEQUEST_GENRE_PACKS": str(tmp_path),
    }
    # --no-warmup keeps this test under ~3s. The race we are guarding
    # against is socket-lifecycle, not model-loading.
    proc = subprocess.Popen(
        [
            "sidequest-renderer",
            "--no-warmup",
            "--output-dir",
            str(tmp_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for the socket file to appear.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if SOCKET_PATH.exists():
            break
        time.sleep(0.1)
    else:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"daemon never created {SOCKET_PATH}\n"
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )
    yield proc
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_socket_file_present_after_bind(daemon_process):
    """The core fail-mode regression: after the daemon logs that it is
    listening, the socket file must actually exist on disk and be a
    socket node — not unlinked out from under the bound fd."""
    assert SOCKET_PATH.exists(), (
        f"daemon bound the socket but the file is missing from {SOCKET_PATH} — "
        "this is the exact failure mode of the 2026-04-26 P1 bug"
    )
    # And it must be a socket, not a regular file.
    assert SOCKET_PATH.is_socket(), (
        f"{SOCKET_PATH} exists but is not a socket node "
        f"(mode={SOCKET_PATH.stat().st_mode:o})"
    )


@pytest.mark.asyncio
async def test_socket_survives_warmup_helper_invocation(daemon_process):
    """If a second invocation of the renderer entry point — for instance a
    warmup helper, a stray ``--shutdown`` racing the live daemon, or any
    future tool that imports ``daemon.py`` — runs while a real daemon is
    bound, the live daemon's socket file MUST remain on disk and clients
    MUST still be able to ``connect()`` to it.

    Without the ``_owns_socket`` guard + ``_live_daemon_pid()`` probe, the
    second invocation's cleanup paths would unlink the live socket and
    leave the system in the exact state described in the bug report:
    process holds bound fd, file gone from disk, clients fail to connect.
    """
    # Simulate the racing helper: invoke ``--shutdown`` against the live
    # daemon, but kill it immediately so it can never actually shutdown
    # cleanly. This exercises the ``send_shutdown`` cleanup branch.
    helper = subprocess.Popen(
        ["sidequest-renderer", "--status"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    helper.wait(timeout=10)

    # The live daemon's socket must still be on disk and connectable.
    assert SOCKET_PATH.exists(), (
        "racing helper unlinked the live daemon's socket — _owns_socket "
        "guard or _live_daemon_pid() probe failed"
    )
    reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
    try:
        request = json.dumps({"id": "lifecycle", "method": "ping", "params": {}})
        writer.write((request + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        response = json.loads(line)
        assert response["id"] == "lifecycle"
        assert response["result"]["status"] == "ok"
    finally:
        writer.close()
        await writer.wait_closed()


def test_send_shutdown_refuses_to_unlink_live_daemons_socket(
    daemon_process,
):
    """``send_shutdown`` must NOT unlink the socket file when the PID file
    points at a live daemon, even if the connect attempt fails. Before the
    fix, any ``ConnectionRefusedError`` (e.g. mid-startup race) would
    unlink the path the live daemon had bound to."""
    # Sanity: the daemon is up and the PID file points at it.
    assert PID_PATH.exists()
    pid = int(PID_PATH.read_text().strip())
    os.kill(pid, 0)  # raises if dead

    # Force ``send_shutdown`` into the cleanup branch by monkey-patching
    # ``open_unix_connection`` to raise ``ConnectionRefusedError`` even
    # though the daemon is alive. This simulates the race in the bug
    # report where the helper saw a transient connect failure.
    import sidequest_daemon.media.daemon as d

    async def _raise_refused(_path):
        raise ConnectionRefusedError("simulated mid-startup race")

    original = asyncio.open_unix_connection
    asyncio.open_unix_connection = _raise_refused  # type: ignore[assignment]
    try:
        asyncio.run(d.send_shutdown())
    finally:
        asyncio.open_unix_connection = original  # type: ignore[assignment]

    # The socket file MUST still be on disk — the live daemon owns it.
    assert SOCKET_PATH.exists(), (
        "send_shutdown unlinked the live daemon's socket despite the "
        "PID file pointing at a running process — the _live_daemon_pid() "
        "guard failed"
    )
    assert PID_PATH.exists(), (
        "send_shutdown unlinked the live daemon's PID file"
    )
