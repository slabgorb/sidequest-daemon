"""MediaPipelineFactory — lazy construction of audio pipelines.

Extracted from Orchestrator._init_audio_pipeline() as part of Epic 58
(Story 58-5). Operates standalone via dependency injection;
no Orchestrator reference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sidequest_daemon.audio.interpreter import AudioInterpreter
from sidequest_daemon.audio.library_backend import LibraryBackend
from sidequest_daemon.audio.mixer import AudioMixer
from sidequest_daemon.audio.queue import AudioQueue
from sidequest_daemon.genre.models import GenrePack

log = logging.getLogger(__name__)


class MediaPipelineFactory:
    """Constructs audio pipelines lazily from injected dependencies."""

    def __init__(
        self,
        genre_pack: GenrePack | None = None,
        audio_base_path: Path | None = None,
    ) -> None:
        self._genre_pack = genre_pack
        self._audio_base_path = audio_base_path

        # Audio pipeline components
        self.audio_mixer: AudioMixer | None = None
        self.audio_backend: LibraryBackend | None = None
        self.audio_interpreter: AudioInterpreter | None = None
        self.music_director: Any = None
        self.audio_queue: AudioQueue | None = None

        # Tracks whether audio was configured (True even if mixer init fails)
        self.audio_was_configured: bool = False

    def init_audio(self) -> None:
        """Initialize audio pipeline (mixer, backend, queue, music director)."""
        audio_config = self._genre_pack.audio if self._genre_pack else None
        has_audio = (
            audio_config is not None
            and self._audio_base_path is not None
            and (audio_config.mood_tracks or audio_config.sfx_library)
        )

        self.audio_was_configured = has_audio

        if not has_audio:
            self.audio_mixer = None
            self.audio_backend = None
            self.audio_interpreter = None
            self.music_director = None
            self.audio_queue = None
            return

        assert audio_config is not None
        assert self._audio_base_path is not None
        mixer_settings = audio_config.mixer

        try:
            self.audio_mixer = AudioMixer(
                duck_level=10 ** (mixer_settings.duck_amount_db / 20),
            )
            log.warning("AUDIO: AudioMixer initialized successfully")
        except Exception as exc:
            log.warning("AUDIO: AudioMixer init FAILED: %s", exc)
            self.audio_mixer = None

        if self.audio_mixer is not None and hasattr(self.audio_mixer, "channels"):
            self.audio_mixer.set_volume("music", mixer_settings.music_volume)
            self.audio_mixer.set_volume("sfx", mixer_settings.sfx_volume)
            self.audio_mixer.crossfade_duration_ms = (
                mixer_settings.crossfade_default_ms
            )

        self.audio_backend = LibraryBackend(audio_config, self._audio_base_path)
        self.audio_interpreter = AudioInterpreter()
        self.music_director = None  # MusicDirector lives in the API, not the daemon
        self.audio_queue = AudioQueue(
            backend=self.audio_backend,
            mixer=self.audio_mixer,
        )
