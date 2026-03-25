"""DaemonClient — connects to sidequest-renderer daemon via Unix domain socket.

Drop-in replacement for MediaWorker: same interface (status, start, send, stop)
so SubprocessRenderer works unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sidequest_daemon.media.protocol import WorkerRequest, WorkerResponse, WorkerStatus
from sidequest_daemon.media.worker import ProtocolError, WorkerCrashedError, WorkerError, WorkerNotReady

logger = logging.getLogger(__name__)


class DaemonClient:
    """Client that talks to the renderer daemon over a Unix domain socket."""

    def __init__(self, socket_path: Path, *, default_timeout: float = 900.0) -> None:
        self._socket_path = socket_path
        self._default_timeout = default_timeout
        self._status = WorkerStatus.IDLE
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._send_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "renderer-daemon"

    @property
    def status(self) -> WorkerStatus:
        return self._status

    async def start(self) -> None:
        """Connect to the daemon socket and verify with a ping."""
        self._status = WorkerStatus.STARTING
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self._socket_path)
            )
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            self._status = WorkerStatus.ERROR
            raise WorkerCrashedError(f"Cannot connect to daemon: {exc}") from exc

        # Verify connection with ping
        try:
            ping = WorkerRequest(method="ping")
            resp = await self._send_raw(ping, timeout=5.0)
            if resp.error is not None:
                raise WorkerError(f"Daemon ping failed: {resp.error.message}")
        except WorkerError:
            self._status = WorkerStatus.ERROR
            self._close()
            raise

        self._status = WorkerStatus.READY

    async def stop(self, *, timeout: float = 5.0) -> None:
        """Disconnect from daemon. Does NOT shut down the daemon itself."""
        self._close()
        self._status = WorkerStatus.STOPPED

    async def health_check(self) -> bool:
        """Ping the daemon."""
        if self._status not in (WorkerStatus.READY, WorkerStatus.BUSY):
            return False
        try:
            ping = WorkerRequest(method="ping")
            resp = await self._send_raw(ping, timeout=5.0)
            return resp.error is None
        except WorkerError:
            return False

    async def send(self, request: WorkerRequest) -> WorkerResponse:
        """Send a request to the daemon and await the response.

        Concurrent calls are serialized via an internal lock so that
        multiple render cues can be queued without being rejected.
        """
        if self._status not in (WorkerStatus.READY, WorkerStatus.BUSY):
            raise WorkerNotReady(f"DaemonClient is {self._status.value}, not READY")

        timeout = request.timeout if request.timeout is not None else self._default_timeout
        async with self._send_lock:
            self._status = WorkerStatus.BUSY
            try:
                resp = await self._send_raw(request, timeout=timeout)
                self._status = WorkerStatus.READY
                return resp
            except WorkerError:
                self._status = WorkerStatus.ERROR
                raise

    async def _send_raw(self, request: WorkerRequest, *, timeout: float) -> WorkerResponse:
        """Write JSON line to socket, read JSON line response."""
        if self._writer is None or self._reader is None:
            raise WorkerCrashedError("Not connected to daemon")

        line = request.model_dump_json() + "\n"
        self._writer.write(line.encode())
        await self._writer.drain()

        try:
            raw = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            raise WorkerCrashedError(
                f"Daemon did not respond within {timeout}s"
            ) from None

        if not raw:
            raise WorkerCrashedError("Daemon closed connection")

        try:
            data = json.loads(raw.decode())
            return WorkerResponse.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            raise ProtocolError(f"Daemon sent unparseable response: {exc}") from exc

    def _close(self) -> None:
        """Close the socket connection."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.exception("Failed to close daemon socket writer")
        self._reader = None
        self._writer = None
