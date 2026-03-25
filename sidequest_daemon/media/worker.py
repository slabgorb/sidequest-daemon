"""MediaWorker — manages a subprocess communicating via JSON-line protocol."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sidequest_daemon.media.protocol import WorkerRequest, WorkerResponse, WorkerStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class WorkerError(Exception):
    """Base exception for all media worker errors."""


class WorkerStartError(WorkerError):
    """Worker subprocess failed to start or respond to initial ping."""


class WorkerTimeoutError(WorkerError):
    """Request exceeded its timeout."""


class WorkerCrashedError(WorkerError):
    """Worker subprocess exited unexpectedly."""


class ProtocolError(WorkerError):
    """Worker sent malformed or unparseable output."""


class WorkerNotReady(WorkerError):
    """Attempted to send a request while worker is not in READY state."""


# ---------------------------------------------------------------------------
# MediaWorker
# ---------------------------------------------------------------------------


class MediaWorker:
    """Manages a subprocess that communicates via JSON-line stdin/stdout."""

    def __init__(
        self,
        *,
        name: str,
        command: list[str],
        workdir: Path,
        default_timeout: float = 30.0,
        startup_timeout: float = 10.0,
    ) -> None:
        self._name = name
        self._command = command
        self._workdir = workdir
        self._default_timeout = default_timeout
        self._startup_timeout = startup_timeout
        self._status = WorkerStatus.IDLE
        self._process: asyncio.subprocess.Process | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def command(self) -> list[str]:
        return list(self._command)

    @property
    def status(self) -> WorkerStatus:
        return self._status

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the subprocess and wait for it to respond to a ping."""
        self._status = WorkerStatus.STARTING
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workdir,
            )
        except (FileNotFoundError, OSError) as exc:
            self._status = WorkerStatus.ERROR
            raise WorkerStartError(
                f"Failed to start worker '{self._name}': {exc}"
            ) from exc

        # Send a ping and wait for response within startup timeout
        try:
            ping = WorkerRequest(method="ping")
            resp = await self._send_raw(ping, timeout=self._startup_timeout)
            if resp.error is not None:
                raise WorkerStartError(
                    f"Worker '{self._name}' ping returned error: {resp.error.message}"
                )
        except (
            WorkerTimeoutError,
            WorkerCrashedError,
            ProtocolError,
            WorkerStartError,
        ) as exc:
            self._status = WorkerStatus.ERROR
            await self._kill()
            if isinstance(exc, WorkerStartError):
                raise
            raise WorkerStartError(
                f"Worker '{self._name}' did not respond to ping: {exc}"
            ) from exc

        self._status = WorkerStatus.READY

    async def stop(self, *, timeout: float = 5.0) -> None:
        """Gracefully stop the subprocess."""
        if self._status in (WorkerStatus.STOPPED, WorkerStatus.IDLE):
            self._status = WorkerStatus.STOPPED
            return

        if self._process is None:
            self._status = WorkerStatus.STOPPED
            return

        # Try graceful shutdown
        try:
            shutdown_req = WorkerRequest(method="shutdown")
            await self._send_raw(shutdown_req, timeout=timeout)
        except (WorkerError, Exception) as exc:
            logger.debug("Graceful shutdown failed for worker '%s': %s", self._name, exc)

        # Wait for process to exit, or force kill
        try:
            await asyncio.wait_for(self._process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            await self._kill()

        self._status = WorkerStatus.STOPPED

    async def health_check(self) -> bool:
        """Send a ping and return True if the worker responds."""
        if self._status not in (WorkerStatus.READY, WorkerStatus.BUSY):
            return False
        try:
            ping = WorkerRequest(method="ping")
            resp = await self._send_raw(ping, timeout=5.0)
            return resp.error is None
        except WorkerError:
            return False

    # -- Send/receive --------------------------------------------------------

    async def send(self, request: WorkerRequest) -> WorkerResponse:
        """Send a request and await the response."""
        if self._status != WorkerStatus.READY:
            raise WorkerNotReady(
                f"Worker '{self._name}' is {self._status.value}, not READY"
            )

        self._status = WorkerStatus.BUSY
        timeout = (
            request.timeout if request.timeout is not None else self._default_timeout
        )
        try:
            resp = await self._send_raw(request, timeout=timeout)
            self._status = WorkerStatus.READY
            return resp
        except WorkerTimeoutError:
            self._status = WorkerStatus.ERROR
            raise
        except WorkerCrashedError:
            self._status = WorkerStatus.ERROR
            raise
        except WorkerError:
            self._status = WorkerStatus.ERROR
            raise

    # -- Internal ------------------------------------------------------------

    async def _send_raw(
        self, request: WorkerRequest, *, timeout: float
    ) -> WorkerResponse:
        """Write request JSON to stdin, read response JSON from stdout."""
        proc = self._process
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise WorkerCrashedError(f"Worker '{self._name}' has no active process")

        if proc.returncode is not None:
            raise WorkerCrashedError(
                f"Worker '{self._name}' exited with code {proc.returncode}"
            )

        line = request.model_dump_json() + "\n"
        proc.stdin.write(line.encode())
        await proc.stdin.drain()

        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            raise WorkerTimeoutError(
                f"Worker '{self._name}' did not respond within {timeout}s"
            ) from None

        if not raw:
            # Empty read means process exited
            raise WorkerCrashedError(
                f"Worker '{self._name}' closed stdout unexpectedly"
            )

        try:
            data = json.loads(raw.decode())
            return WorkerResponse.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            raise ProtocolError(
                f"Worker '{self._name}' sent unparseable response: {exc}"
            ) from exc

    async def _kill(self) -> None:
        """Force-kill the subprocess."""
        if self._process is not None and self._process.returncode is None:
            self._process.kill()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
