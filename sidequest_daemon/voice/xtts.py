"""XTTSEngine — XTTS v2 zero-shot voice cloning SynthesisEngine."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from sidequest_daemon.voice.errors import ModelLoadError, SynthesisFailedError
from sidequest_daemon.voice.protocol import (
    AudioSegment,
    SynthesisEngine,
    SynthesisMode,
    VoicePreset,
)


class VoiceReference(BaseModel):
    """Reference audio sample for zero-shot voice cloning."""

    character_id: str
    audio_path: Path
    file_size_bytes: int


class VoiceReferenceManager:
    """Load, cache, and query voice reference audio files for XTTS cloning."""

    def __init__(self) -> None:
        self._references: dict[str, VoiceReference] = {}

    def load_reference(self, character_id: str, audio_path: Path) -> VoiceReference:
        """Load a voice reference from an audio file.

        Raises FileNotFoundError if the path does not exist.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        ref = VoiceReference(
            character_id=character_id,
            audio_path=audio_path,
            file_size_bytes=audio_path.stat().st_size,
        )
        self._references[character_id] = ref
        return ref

    def get_reference(self, character_id: str) -> VoiceReference | None:
        """Return cached reference for a character, or None."""
        return self._references.get(character_id)

    def list_characters(self) -> list[str]:
        """Return all character IDs with loaded references."""
        return list(self._references.keys())

    def remove_reference(self, character_id: str) -> None:
        """Remove a cached reference. No-op if not found."""
        self._references.pop(character_id, None)


class XTTSEngine(SynthesisEngine):
    """Zero-shot voice cloning engine backed by XTTS v2."""

    SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        device: str = "cpu",
        voice_references: VoiceReferenceManager | None = None,
    ) -> None:
        self._device = device
        self._is_ready = False
        self._voice_references = voice_references or VoiceReferenceManager()

    # -- SynthesisEngine interface -------------------------------------------

    @property
    def name(self) -> str:
        return "xtts"

    @property
    def supported_modes(self) -> list[SynthesisMode]:
        return [SynthesisMode.BATCH, SynthesisMode.STREAMING]

    async def synthesize(self, text: str, voice_preset: VoicePreset) -> AudioSegment:
        if not self._is_ready:
            raise ModelLoadError("Engine not ready — call warm_up() first")

        normalized = self._normalize_text(text)
        if not normalized:
            return AudioSegment(data=b"", sample_rate=self.SAMPLE_RATE, channels=1)

        try:
            raw = self._synthesize_cloned(normalized, voice_preset=voice_preset)
        except (ModelLoadError, SynthesisFailedError):
            raise
        except Exception as exc:
            raise SynthesisFailedError(str(exc)) from exc

        return AudioSegment(data=raw, sample_rate=self.SAMPLE_RATE, channels=1)

    async def warm_up(self) -> None:
        self._load_model()
        self._is_ready = True

    async def shutdown(self) -> None:
        self._is_ready = False

    # -- Properties ----------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def voice_references(self) -> VoiceReferenceManager:
        return self._voice_references

    # -- Internal ------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the XTTS v2 model. Overridden in tests via patch."""

    def _synthesize_cloned(self, text: str, *, voice_preset: VoicePreset) -> bytes:
        """Run XTTS synthesis with voice cloning. Overridden in tests via patch."""
        return b""

    def _normalize_text(self, text: str) -> str:
        """Normalize input text: strip, collapse whitespace, remove newlines."""
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
