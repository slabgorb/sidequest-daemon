"""PiperEngine — concrete SynthesisEngine using Piper TTS."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from sidequest_daemon.voice.errors import ModelLoadError, SynthesisFailedError
from sidequest_daemon.voice.protocol import (
    AudioSegment,
    SynthesisEngine,
    SynthesisMode,
    VoicePreset,
)

if TYPE_CHECKING:
    from sidequest_daemon.voice.model_manager import PiperModelManager


class PiperEngine(SynthesisEngine):
    """Text-to-speech engine backed by Piper TTS."""

    DEFAULT_MODEL = "en_US-lessac-medium"
    SAMPLE_RATE = 22050

    def __init__(
        self,
        *,
        model: str | None = None,
        model_manager: PiperModelManager | None = None,
    ) -> None:
        self._default_model = model or self.DEFAULT_MODEL
        self._model_manager = model_manager
        self._is_ready = False

    # -- SynthesisEngine interface -------------------------------------------

    @property
    def name(self) -> str:
        return "piper"

    @property
    def supported_modes(self) -> list[SynthesisMode]:
        return [SynthesisMode.BATCH]

    async def synthesize(self, text: str, voice_preset: VoicePreset) -> AudioSegment:
        if not self._is_ready:
            raise ModelLoadError("Engine not ready — call warm_up() first")

        normalized = self._normalize_text(text)
        if not normalized:
            return AudioSegment(data=b"", sample_rate=self.SAMPLE_RATE, channels=1)

        model = self._select_model(voice_preset)
        try:
            raw = self._synthesize_raw(normalized, model=model)
        except (ModelLoadError, SynthesisFailedError):
            raise
        except Exception as exc:
            raise SynthesisFailedError(str(exc)) from exc

        return AudioSegment(data=raw, sample_rate=self.SAMPLE_RATE, channels=1)

    async def warm_up(self) -> None:
        if self._model_manager is not None:
            await self._model_manager.ensure_model(self._default_model)
        self._load_model()
        self._is_ready = True

    def resolved_model_path(self, name: str) -> "Path":
        """Resolve model path through the model manager."""
        if self._model_manager is None:
            raise ModelLoadError("No model manager configured")
        return self._model_manager.model_path(name)

    async def shutdown(self) -> None:
        self._is_ready = False

    # -- Properties ----------------------------------------------------------

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # -- Internal ------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Piper model."""
        model_dir = Path.home() / ".sidequest" / "models" / "piper"
        model_path = model_dir / f"{self._default_model}.onnx"
        if not model_path.exists():
            return
        try:
            from piper import PiperVoice
            self._voice = PiperVoice.load(str(model_path))
        except Exception:
            self._voice = None

    def _synthesize_raw(self, text: str, *, model: str) -> bytes:
        """Run Piper synthesis and return raw PCM s16le bytes."""
        import numpy as np

        voice = getattr(self, "_voice", None)
        if voice is None:
            return b""

        pcm_chunks = []
        for chunk in voice.synthesize(text):
            # Convert float32 [-1,1] to int16 PCM
            int16 = (np.clip(chunk.audio_float_array, -1.0, 1.0) * 32767).astype(np.int16)
            pcm_chunks.append(int16.tobytes())

        return b"".join(pcm_chunks)

    def _normalize_text(self, text: str) -> str:
        """Normalize input text for TTS: strip, collapse whitespace, remove newlines."""
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _select_model(self, preset: VoicePreset) -> str:
        """Pick the model: preset override > engine default."""
        return preset.model if preset.model is not None else self._default_model
