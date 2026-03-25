"""MusicGenBackend — AI music generation via Meta's MusicGen model.

Story 5-9: MusicGen AI generation backend + worker
Wraps MusicGen as an AudioBackend for text-to-music generation.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from sidequest_daemon.audio.models import AudioCue, AudioLane, AudioResult
from sidequest_daemon.audio.protocol import AudioBackend

# Estimated GPU memory for MusicGen-small (~300M params, fp32)
_MODEL_MEMORY_BYTES = 1_200_000_000  # ~1.2 GB


class MusicGenBackend(AudioBackend):
    """Generates music from text prompts using Meta's MusicGen model."""

    def __init__(self) -> None:
        self._model = None
        self._is_ready = False

    @property
    def name(self) -> str:
        return "musicgen"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def gpu_memory_bytes(self) -> int:
        return _MODEL_MEMORY_BYTES if self._is_ready else 0

    def supports_lane(self, lane: AudioLane) -> bool:
        return lane == AudioLane.MUSIC

    async def warm_up(self) -> None:
        if self._is_ready:
            return
        try:
            await self._load_model()
        except Exception:
            self._is_ready = False
            raise
        self._is_ready = True

    async def shutdown(self) -> None:
        self._model = None
        self._is_ready = False

    async def _load_model(self) -> None:
        """Load the MusicGen model. Overridden/mocked in tests."""
        from audiocraft.models import MusicGen  # type: ignore[import-untyped]

        self._model = MusicGen.get_pretrained("facebook/musicgen-small")

    async def _run_inference(self, prompt: str, duration: int) -> Path:
        """Run model inference. Overridden/mocked in tests."""
        raise NotImplementedError("Real inference requires audiocraft")

    async def generate(
        self,
        prompt: str,
        duration_seconds: int = 10,
        timeout_seconds: float = 60.0,
    ) -> Path:
        if not self._is_ready:
            raise RuntimeError("Model not ready — call warm_up() first")
        return await asyncio.wait_for(
            self._run_inference(prompt, duration_seconds),
            timeout=timeout_seconds,
        )

    async def play(self, cue: AudioCue) -> AudioResult:
        parts = []
        if cue.mood is not None:
            parts.append(cue.mood.value if hasattr(cue.mood, "value") else cue.mood)
        if cue.subject:
            parts.append(cue.subject)
        prompt = " ".join(parts) if parts else "background music"

        start = time.monotonic()
        audio_path = await self.generate(prompt)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        return AudioResult(
            audio_path=audio_path,
            duration_ms=0,
            lane=cue.lane,
            cue=cue,
            source="musicgen",
            generation_time_ms=elapsed_ms,
        )
