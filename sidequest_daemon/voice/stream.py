"""Async synthesis stream — concurrent streaming TTS with cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from sidequest_daemon.voice.protocol import AudioSegment, SynthesisEngine, VoicePreset
from sidequest_daemon.voice.segmenter import SentenceSegmenter


_LOOKAHEAD = 3  # max concurrent synthesis tasks


class SynthesisStream:
    """Streams synthesis results as an async iterator of (text, audio, preset) tuples.

    Launches concurrent synthesis tasks and yields results in input order.
    Supports cancellation to interrupt in-flight synthesis.
    """

    def __init__(
        self,
        engine: SynthesisEngine,
        effects_library: Any | None = None,
    ) -> None:
        self._engine = engine
        self._effects_library = effects_library
        self._cancelled = False
        self._tasks: list[asyncio.Task[AudioSegment]] = []
        self._segmenter = SentenceSegmenter()

    async def warm_up(self) -> None:
        """Warm up the underlying synthesis engine."""
        await self._engine.warm_up()

    def cancel(self) -> None:
        """Cancel in-flight synthesis. Idempotent."""
        self._cancelled = True
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def _apply_effects(self, audio: AudioSegment, preset: VoicePreset) -> AudioSegment:
        """Apply effects from the library based on the preset's effects list."""
        if not self._effects_library or not preset.effects:
            return audio

        from sidequest_daemon.voice.effects import PedalboardEffectsChain

        # Look up and compose effects; unknown names → passthrough
        known = [n for n in preset.effects if n in self._effects_library._presets]
        if not known:
            return audio

        config = self._effects_library.compose_presets(known)
        if not config.get("effects"):
            return audio

        chain = PedalboardEffectsChain.from_dict(config)

        # Convert PCM s16le bytes → float32 numpy, process, convert back
        pcm = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32767.0
        processed = chain.process(pcm, sample_rate=audio.sample_rate)
        out_bytes = (np.clip(processed, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        return AudioSegment(
            data=out_bytes,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            format=audio.format,
        )

    async def synthesize(
        self, pairs: list[tuple[str, VoicePreset]]
    ) -> AsyncIterator[tuple[str, AudioSegment, VoicePreset]]:
        """Synthesize pairs concurrently, yielding (text, audio, preset) in order."""
        if not pairs:
            return

        self._cancelled = False
        self._tasks = []

        # Pre-launch up to _LOOKAHEAD tasks.
        next_launch = 0
        for i in range(min(_LOOKAHEAD, len(pairs))):
            text, preset = pairs[i]
            self._tasks.append(
                asyncio.create_task(self._engine.synthesize(text, preset))
            )
            next_launch = i + 1

        try:
            for i, (text, preset) in enumerate(pairs):
                if self._cancelled:
                    break
                try:
                    audio = await self._tasks[i]
                except asyncio.CancelledError:
                    break
                if self._cancelled:
                    break

                # Launch the next task before yielding so synthesis stays ahead.
                if next_launch < len(pairs):
                    t, p = pairs[next_launch]
                    self._tasks.append(
                        asyncio.create_task(self._engine.synthesize(t, p))
                    )
                    next_launch += 1

                audio = self._apply_effects(audio, preset)
                yield text, audio, preset
        finally:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            self._tasks = []

    async def synthesize_text(
        self, text: str, preset: VoicePreset
    ) -> AsyncIterator[tuple[str, AudioSegment, VoicePreset]]:
        """Segment text into sentences, then synthesize each one.

        Composes SentenceSegmenter with the synthesis stream.
        """
        sentences = self._segmenter.segment(text)
        pairs = [(sentence, preset) for sentence in sentences]
        async for item in self.synthesize(pairs):
            yield item
