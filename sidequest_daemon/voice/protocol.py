"""Voice synthesis protocol models and SynthesisEngine ABC.

Defines the abstract base class for synthesis engines, plus the data models
for voice presets, audio segments, and synthesis modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


class SynthesisMode(str, Enum):
    """Synthesis execution modes."""

    BATCH = "batch"
    STREAMING = "streaming"


class VoicePreset(BaseModel):
    """Configuration for a character's voice — pitch, rate, effects, model."""

    name: str
    pitch: float = 1.0
    rate: float = 1.0
    effects: list[str | dict] = []
    model: str | None = None
    voice_id: int | None = None


class AudioSegment(BaseModel):
    """Synthesized audio data with metadata."""

    data: bytes
    sample_rate: int
    channels: int
    format: str = "pcm_s16le"

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds, computed from data size and format.

        Assumes 16-bit (2 bytes per sample) PCM audio.
        """
        if not self.data:
            return 0
        bytes_per_sample = 2  # 16-bit
        total_samples = len(self.data) // (bytes_per_sample * self.channels)
        return (total_samples / self.sample_rate) * 1000


class SynthesisEngine(ABC):
    """Abstract base for text-to-speech synthesis backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier (e.g. 'piper', 'xtts')."""
        ...

    @property
    @abstractmethod
    def supported_modes(self) -> list[SynthesisMode]:
        """Which synthesis modes this engine supports."""
        ...

    @abstractmethod
    async def synthesize(self, text: str, voice_preset: VoicePreset) -> AudioSegment:
        """Synthesize text to audio using the given voice preset."""
        ...

    @abstractmethod
    async def warm_up(self) -> None:
        """Pre-load models, allocate resources. Called once at startup."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources."""
        ...
