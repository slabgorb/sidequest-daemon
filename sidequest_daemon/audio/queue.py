"""AudioQueue — async background audio processing.

Story 5-5: Async bridge between AudioInterpreter and AudioMixer/LibraryBackend.
Receives AudioCues, resolves via LibraryBackend, schedules playback through AudioMixer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sidequest_daemon.audio.models import AudioCue, AudioLane

logger = logging.getLogger(__name__)

_DRAIN_TIMEOUT_S = 2.0


class AudioQueue:
    """Async background queue that resolves AudioCues and routes to AudioMixer."""

    def __init__(
        self, *, backend: Any, mixer: Any, voice_backend: Any | None = None
    ) -> None:
        self._backend = backend
        self._mixer = mixer
        self._voice_backend = voice_backend
        self._queue: asyncio.PriorityQueue[tuple[int, int, AudioCue]] = (
            asyncio.PriorityQueue()
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._seq = 0
        self.current_mood = None

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self.is_running:
            return
        self._stopping = False
        self._queue = asyncio.PriorityQueue()
        self._seq = 0
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if not self.is_running:
            return
        self._stopping = True
        # Sentinel to unblock the worker
        self._queue.put_nowait((999_999, 999_999, None))  # type: ignore[arg-type]
        try:
            await asyncio.wait_for(self._worker_task, timeout=_DRAIN_TIMEOUT_S)  # type: ignore[arg-type]
        except asyncio.TimeoutError:
            self._worker_task.cancel()  # type: ignore[union-attr]
        self._worker_task = None

    async def enqueue(self, cue: AudioCue) -> None:
        self._seq += 1
        # Negate priority so higher priority values are processed first
        self._queue.put_nowait((-cue.priority, self._seq, cue))

    async def _worker(self) -> None:
        while True:
            try:
                _, _, cue = await self._queue.get()
            except asyncio.CancelledError:
                return

            if cue is None:
                # Sentinel: drain remaining items then exit
                while not self._queue.empty():
                    try:
                        _, _, remaining = self._queue.get_nowait()
                        if remaining is not None:
                            self._process_cue(remaining)
                    except asyncio.QueueEmpty:
                        break
                return

            self._process_cue(cue)

    def _process_cue(self, cue: AudioCue) -> None:
        logger.warning("AUDIO: processing cue lane=%s mood=%s mixer=%s voice=%s",
                       cue.lane, cue.mood, self._mixer is not None, self._voice_backend is not None)
        # FADE_OUT cues have no track to resolve — route directly to mixer.stop
        if cue.metadata.get("fade_out") and self._mixer is not None:
            channel = cue.lane.value
            try:
                self._mixer.stop(channel, fade_out_ms=cue.fade_out_ms)
            except Exception:
                logger.debug("Mixer fade-out error for cue %s", cue, exc_info=True)
            return

        try:
            path = self._backend.resolve(cue)
        except Exception:
            logger.debug("Failed to resolve cue %s", cue, exc_info=True)
            return

        if path is None:
            return

        # Set resolved_path in metadata so voice adapter can find the file
        cue.metadata["resolved_path"] = str(path)

        channel = cue.lane.value

        if cue.lane == AudioLane.MUSIC:
            self.current_mood = cue.mood

        # Dispatch to voice backend (Discord) if available
        if self._voice_backend is not None:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._voice_backend.play(cue))
                else:
                    loop.run_until_complete(self._voice_backend.play(cue))
            except Exception:
                logger.debug("Voice backend error for cue %s", cue, exc_info=True)

        if self._mixer is None:
            return

        try:
            if cue.lane == AudioLane.MUSIC and self._mixer.is_playing("music"):
                self._mixer.crossfade(
                    channel=channel,
                    path=path,
                    duration_ms=cue.crossfade_ms,
                )
            else:
                # Never loop music — play once, then re-resolve a new track
                play_kwargs: dict = {
                    "channel": channel,
                    "path": path,
                    "loop": False,
                }
                if cue.fade_in_ms > 0:
                    play_kwargs["fade_in_ms"] = cue.fade_in_ms
                self._mixer.play(**play_kwargs)

                # Schedule a follow-up track when this one ends
                if cue.lane == AudioLane.MUSIC:
                    self._schedule_next_track(cue)

            if cue.lane == AudioLane.SFX:
                self._mixer.notify_channel_done("sfx")
        except Exception:
            logger.debug("Mixer error for cue %s", cue, exc_info=True)

    def _schedule_next_track(self, cue: AudioCue) -> None:
        """Poll for music channel idle, then enqueue a follow-up track."""
        import asyncio

        async def _poll_and_enqueue() -> None:
            # Wait for current track to finish
            while self._mixer is not None and self._mixer.is_playing("music"):
                await asyncio.sleep(1.0)
            if self._stopping:
                return
            # Re-enqueue with same mood so the rotator picks a different track
            next_cue = AudioCue(
                lane=AudioLane.MUSIC,
                mood=cue.mood,
                intensity=cue.intensity,
                priority=cue.priority,
                crossfade_ms=cue.crossfade_ms,
                fade_in_ms=cue.fade_in_ms,
            )
            await self.enqueue(next_cue)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_poll_and_enqueue())
        except Exception:
            logger.debug("Failed to schedule next track", exc_info=True)
