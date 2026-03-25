"""TTS (text-to-speech) worker for the unified daemon.

Story 23-13: Wraps a TTS model (kokoro or similar) as a daemon worker with
the same interface as FluxWorker (load_model, warm_up, render, cleanup).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path


class TTSWorker:
    """Text-to-speech worker for narrator voice synthesis."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def load_model(self) -> None:
        """Load the TTS model."""
        # Placeholder — will wire kokoro or similar ~100MB model
        # For now, mark as loaded so the daemon routing works
        self.model = "tts-placeholder"

    def warm_up(self) -> dict:
        """Warm up the TTS model. Returns timing metadata."""
        start = time.monotonic()
        # TTS models are small — warmup is just loading
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {"warmup_ms": elapsed_ms}

    def render(self, params: dict) -> dict:
        """Synthesize speech from params. Returns result dict with audio_path."""
        text = params.get("text", "")
        voice = params.get("voice", "narrator")

        output_name = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = self.output_dir / output_name

        start = time.monotonic()
        # Placeholder inference — real implementation will call kokoro
        output_path.write_bytes(b"")  # empty file as stub
        elapsed_ms = int((time.monotonic() - start) * 1000)

        return {
            "audio_path": str(output_path),
            "duration_ms": 0,
            "elapsed_ms": elapsed_ms,
            "voice": voice,
        }

    def cleanup(self) -> None:
        """Release the model."""
        self.model = None
