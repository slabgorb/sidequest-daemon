"""LocalVoiceAdapter — TUI-side voice adapter for local audio playback.

Streams TTS audio segments to the local audio device (e.g. sounddevice).
Implements the same interface as the Discord voice adapter so the Orchestrator
can use either one interchangeably.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidequest_daemon.audio.models import AudioCue, AudioResult

logger = logging.getLogger(__name__)


class LocalVoiceAdapter:
    """Plays synthesized voice audio through the local audio output device."""

    def __init__(self, *, on_segment: Callable[[int], None] | None = None) -> None:
        self._connected = False
        self._degraded = False
        self._tts_state = "idle"
        self._on_segment = on_segment
        self._ducking_active = False
        self._tts_volume: float = 1.0

    # -- Identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return "local"

    # -- Connection lifecycle -------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    async def warm_up(self) -> None:
        """Prepare the local audio output device."""
        try:
            self._open_audio_device()
            self._connected = True
            self._degraded = False
        except OSError:
            self._connected = False
            self._degraded = True

    async def shutdown(self) -> None:
        """Release audio resources. Safe to call multiple times."""
        self._connected = False
        self._tts_state = "idle"

    # -- TTS state ------------------------------------------------------------

    @property
    def tts_state(self) -> str:
        return self._tts_state

    # -- Ducking --------------------------------------------------------------

    @property
    def ducking_active(self) -> bool:
        return self._ducking_active

    def set_ducking(self, active: bool) -> None:
        self._ducking_active = active

    # -- Streaming ------------------------------------------------------------

    def supports_lane(self, lane: str) -> bool:
        """Check whether this adapter supports the given audio lane."""
        return lane.lower() in ("voice", "music", "sfx")

    async def play(self, cue: AudioCue) -> AudioResult:
        """Play an audio cue (music/sfx).

        For local TUI mode the AudioMixer (pygame) handles actual playback.
        This method exists so the AudioQueue voice_backend call does not error.
        Returns a stub AudioResult.
        """
        from sidequest_daemon.audio.models import AudioLane, AudioResult

        resolved = cue.metadata.get("resolved_path")
        audio_path = Path(resolved) if resolved else Path("/dev/null")

        return AudioResult(
            audio_path=audio_path,
            duration_ms=0,
            lane=cue.lane,
            cue=cue,
            source="local",
        )

    async def stream_voice(self, segments: AsyncIterator) -> None:
        """Stream audio segments to the local output device via pygame.

        Raises RuntimeError if not connected via warm_up().
        """
        if not self._connected:
            raise RuntimeError("Not connected — call warm_up() first")

        self._tts_state = "buffering"
        index = 0
        try:
            async for segment in segments:
                if not self._connected:
                    break
                if not segment.data:
                    continue
                self._tts_state = "streaming"
                self._last_segment_duration_s = 0.0
                self._play_pcm(segment)
                # Non-blocking wait for playback to finish before next segment
                await asyncio.sleep(self._last_segment_duration_s)
                if self._on_segment is not None:
                    self._on_segment(index)
                index += 1
        finally:
            self._tts_state = "idle"

    # -- Internal -------------------------------------------------------------

    def _play_pcm(self, segment: object) -> None:
        """Play a PCM audio segment through pygame mixer."""
        import io
        import struct

        try:
            import pygame
        except ImportError:
            return

        data = getattr(segment, "data", b"")
        sample_rate = getattr(segment, "sample_rate", 22050)
        channels = getattr(segment, "channels", 1)
        if not data:
            return

        # Build a WAV in memory so pygame.mixer.Sound can load it
        num_samples = len(data) // 2  # 16-bit = 2 bytes per sample
        wav_buf = io.BytesIO()
        # WAV header
        data_size = len(data)
        wav_buf.write(b"RIFF")
        wav_buf.write(struct.pack("<I", 36 + data_size))
        wav_buf.write(b"WAVE")
        wav_buf.write(b"fmt ")
        wav_buf.write(struct.pack("<I", 16))  # chunk size
        wav_buf.write(struct.pack("<H", 1))   # PCM format
        wav_buf.write(struct.pack("<H", channels))
        wav_buf.write(struct.pack("<I", sample_rate))
        wav_buf.write(struct.pack("<I", sample_rate * channels * 2))  # byte rate
        wav_buf.write(struct.pack("<H", channels * 2))  # block align
        wav_buf.write(struct.pack("<H", 16))  # bits per sample
        wav_buf.write(b"data")
        wav_buf.write(struct.pack("<I", data_size))
        wav_buf.write(data)
        wav_buf.seek(0)

        try:
            sound = pygame.mixer.Sound(wav_buf)
            # Play on dedicated TTS channel — never steal from music/sfx
            tts_ch = pygame.mixer.Channel(4)  # Reserved TTS channel
            tts_ch.set_volume(self._tts_volume)
            tts_ch.play(sound)
            # Playback duration tracked for non-blocking wait in stream_voice
            # num_samples is total samples across all channels, so divide by
            # sample_rate alone (not sample_rate * channels)
            self._last_segment_duration_s = num_samples / sample_rate
        except Exception:
            logger.debug("Failed to play TTS segment via pygame", exc_info=True)

    def _open_audio_device(self) -> None:
        """Verify pygame mixer is available for TTS playback."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1)
        except Exception:
            raise OSError("pygame mixer not available")
