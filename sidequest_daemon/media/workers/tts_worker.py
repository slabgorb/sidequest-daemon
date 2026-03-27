"""TTS (text-to-speech) worker for the unified daemon.

Story 23-13: Wraps KokoroEngine as a daemon worker with the same interface
as FluxWorker (load_model, warm_up, render, cleanup).

Output: raw PCM s16le at 24000Hz, returned as audio_bytes (list[int]) and
duration_ms for JSON-RPC compatibility with the Rust DaemonSynthesizer.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

from sidequest_daemon.voice.kokoro import KokoroEngine
from sidequest_daemon.voice.protocol import VoicePreset

log = logging.getLogger(__name__)


class TTSWorker:
    """Text-to-speech worker backed by KokoroEngine."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._engine: KokoroEngine | None = None

    def load_model(self) -> None:
        """Load the Kokoro TTS engine."""
        self._engine = KokoroEngine()

    def warm_up(self) -> dict:
        """Warm up the TTS engine. Returns timing metadata."""
        start = time.monotonic()
        if self._engine is not None:
            asyncio.get_event_loop().run_until_complete(self._engine.warm_up())
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {"warmup_ms": elapsed_ms}

    def render(self, params: dict) -> dict:
        """Synthesize speech from params.

        Returns dict with:
          - audio_bytes: list[int] — raw PCM s16le bytes as JSON-safe integers
          - duration_ms: int — audio duration in milliseconds
          - voice: str — voice name used
          - elapsed_ms: int — wall-clock synthesis time
        """
        text = params.get("text", "")
        voice_name = params.get("voice", params.get("voice_id", "narrator"))
        voice_id = None
        speed = params.get("speed", 1.0)

        # Resolve voice_id: if it's numeric, use it directly
        try:
            voice_id = int(voice_name)
            preset_name = f"voice_{voice_id}"
        except (ValueError, TypeError):
            preset_name = str(voice_name)

        preset = VoicePreset(
            name=preset_name,
            rate=float(speed),
            voice_id=voice_id,
        )

        start = time.monotonic()

        if self._engine is None:
            raise RuntimeError("TTS engine not loaded — call load_model() first")

        loop = asyncio.get_event_loop()
        segment = loop.run_until_complete(self._engine.synthesize(text, preset))
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Optionally write to file for debugging / caching
        output_name = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = self.output_dir / output_name
        output_path.write_bytes(segment.data)

        # Return audio_bytes as list[int] for JSON serialization.
        # Rust serde deserializes Vec<u8> from a JSON array of integers.
        return {
            "audio_bytes": list(segment.data),
            "duration_ms": int(segment.duration_ms),
            "elapsed_ms": elapsed_ms,
            "voice": preset_name,
            "audio_path": str(output_path),
        }

    def cleanup(self) -> None:
        """Release the engine."""
        if self._engine is not None:
            asyncio.get_event_loop().run_until_complete(self._engine.shutdown())
            self._engine = None
