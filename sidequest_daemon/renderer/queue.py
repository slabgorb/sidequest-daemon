"""Async queue that processes StageCues through a Renderer in background."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

from sidequest_daemon.renderer.base import Renderer
from sidequest_daemon.renderer.models import RenderResult, StageCue

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidequest_daemon.media.cache import SceneCache


class RenderQueue:
    """Async queue that processes StageCues through a Renderer in background."""

    def __init__(
        self,
        renderer: Renderer,
        on_complete: Callable[[RenderResult], Awaitable[None]],
    ) -> None:
        self._renderer = renderer
        self._on_complete = on_complete
        self._queue: asyncio.Queue[tuple[StageCue, asyncio.Future[RenderResult | None]]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._current_turn_id: int = 0

    @property
    def current_turn_id(self) -> int:
        """Current turn id for stale-cue filtering."""
        return self._current_turn_id

    def advance_turn(self, turn_id: int) -> None:
        """Advance to a new turn — cues from older turns will be skipped."""
        self._current_turn_id = turn_id

    async def start(self) -> None:
        """Start the background worker."""
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Drain queue and stop the worker."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def submit(self, cue: StageCue) -> asyncio.Future[RenderResult | None]:
        """Enqueue a render request and return a Future that resolves on completion."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RenderResult | None] = loop.create_future()
        await self._queue.put((cue, future))
        return future

    async def flush(self) -> None:
        """Discard all pending (not-yet-processing) cues from the queue."""
        discarded: list[tuple[StageCue, asyncio.Future[RenderResult | None]]] = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                discarded.append(item)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        for _cue, fut in discarded:
            if not fut.done():
                fut.set_result(None)

    async def _worker(self) -> None:
        """Process cues one at a time (GPU is the bottleneck)."""
        while True:
            cue, future = await self._queue.get()
            # Skip stale cues from previous turns
            if cue.turn_id < self._current_turn_id:
                log.info("RenderQueue skipping stale cue: tier=%s turn_id=%d (current=%d)", cue.tier, cue.turn_id, self._current_turn_id)
                if not future.done():
                    future.set_result(None)
                self._queue.task_done()
                continue
            if not self._renderer.supports_tier(cue.tier):
                log.warning("RenderQueue skipping unsupported tier: %s", cue.tier)
                if not future.done():
                    future.set_result(None)
                self._queue.task_done()
                continue
            log.info("RenderQueue processing cue: tier=%s subject=%s", cue.tier, cue.subject)
            try:
                result = await self._renderer.render(cue)
                # Suppress on_complete for results that became stale during rendering
                if cue.turn_id < self._current_turn_id:
                    log.info("RenderQueue suppressing stale result: tier=%s turn_id=%d", cue.tier, cue.turn_id)
                    if not future.done():
                        future.set_result(None)
                else:
                    log.info("RenderQueue cue complete: tier=%s path=%s", cue.tier, result.image_path)
                    await self._on_complete(result)
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                log.error("RENDER_FAILED: tier=%s subject=%r error=%s (%s)",
                          cue.tier, cue.subject[:60] if cue.subject else "", type(exc).__name__, exc)
                if not future.done():
                    future.set_result(None)
            finally:
                self._queue.task_done()


class CachedRenderQueue:
    """RenderQueue wrapper that checks SceneCache before rendering."""

    def __init__(
        self,
        renderer: Renderer,
        cache: SceneCache,
        on_complete: Callable[[RenderResult], Awaitable[None]],
    ) -> None:
        self._renderer = renderer
        self._cache = cache
        self._on_complete = on_complete
        self._queue: asyncio.Queue[StageCue] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def submit(self, cue: StageCue) -> None:
        await self._queue.put(cue)

    async def _worker(self) -> None:
        while True:
            cue = await self._queue.get()
            try:
                cached = self._cache.get(cue)
                if cached is not None:
                    log.info("RENDER_CACHE: hit for tier=%s subject=%r", cue.tier, cue.subject[:40] if cue.subject else "")
                    await self._on_complete(cached)
                else:
                    log.info("RENDER_CACHE: miss for tier=%s, rendering...", cue.tier)
                    result = await self._renderer.render(cue)
                    self._cache.put(cue, result)
                    await self._on_complete(result)
            except Exception as exc:
                log.error("RENDER_FAILED: tier=%s subject=%r error=%s (%s)",
                          cue.tier, cue.subject[:60] if cue.subject else "", type(exc).__name__, exc)
            finally:
                self._queue.task_done()
