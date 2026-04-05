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


def _run_async(coro):
    """Run an async coroutine from a synchronous thread context.

    Python 3.14 removed the implicit event loop creation in
    asyncio.get_event_loop() for non-main threads. This helper
    creates a fresh event loop, runs the coroutine, and cleans up.
    Safe to call from asyncio.to_thread() worker threads.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
            _run_async(self._engine.warm_up())
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
        pitch = params.get("pitch", 1.0)
        effects = params.get("effects", [])

        # Resolve voice_id: if it's numeric, use it directly
        try:
            voice_id = int(voice_name)
            preset_name = f"voice_{voice_id}"
        except (ValueError, TypeError):
            preset_name = str(voice_name)

        preset = VoicePreset(
            name=preset_name,
            rate=float(speed),
            pitch=float(pitch),
            effects=effects,
            voice_id=voice_id,
        )

        start = time.monotonic()

        if self._engine is None:
            raise RuntimeError("TTS engine not loaded — call load_model() first")

        segment = _run_async(self._engine.synthesize(text, preset))

        # Apply pitch shift and audio effects from creature_voice_presets.
        # Pitch != 1.0 is applied as a PitchShift effect prepended to the chain.
        if pitch != 1.0 or effects:
            import math

            import numpy as np

            from sidequest_daemon.voice.effects import PedalboardEffectsChain

            effect_defs = list(effects)
            if pitch != 1.0:
                semitones = 12.0 * math.log2(pitch)
                effect_defs.insert(0, {"type": "pitch_shift", "params": {"semitones": semitones}})

            chain = PedalboardEffectsChain.from_effect_list(effect_defs)
            pcm = np.frombuffer(segment.data, dtype=np.int16).astype(np.float32) / 32768.0
            processed = chain.process(pcm, sample_rate=segment.sample_rate)
            processed = np.clip(processed, -1.0, 1.0)
            segment_data = (processed * 32767).astype(np.int16).tobytes()
            segment = type(segment)(
                data=segment_data,
                sample_rate=segment.sample_rate,
                channels=segment.channels,
                format=segment.format,
            )

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
            _run_async(self._engine.shutdown())
            self._engine = None
