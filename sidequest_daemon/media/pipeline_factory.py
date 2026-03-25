"""MediaPipelineFactory — lazy construction of audio and voice pipelines.

Extracted from Orchestrator._init_audio_pipeline() and _init_voice_pipeline()
as part of Epic 58 (Story 58-5). Operates standalone via dependency injection;
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
from sidequest_daemon.voice.presets import EffectsPresetLibrary
from sidequest_daemon.voice.registry import VoicePresetRegistry
from sidequest_daemon.voice.router import VoiceRouter

log = logging.getLogger(__name__)


class MediaPipelineFactory:
    """Constructs audio and voice pipelines lazily from injected dependencies."""

    def __init__(
        self,
        genre_pack: GenrePack | None = None,
        audio_base_path: Path | None = None,
        voice_adapter: Any | None = None,
        enable_tts: bool = True,
    ) -> None:
        self._genre_pack = genre_pack
        self._audio_base_path = audio_base_path
        self._voice_adapter = voice_adapter
        self._enable_tts = enable_tts

        # Audio pipeline components
        self.audio_mixer: AudioMixer | None = None
        self.audio_backend: LibraryBackend | None = None
        self.audio_interpreter: AudioInterpreter | None = None
        self.music_director: Any = None
        self.audio_queue: AudioQueue | None = None
        self.effects_library: EffectsPresetLibrary | None = None

        # Tracks whether audio was configured (True even if mixer init fails)
        self.audio_was_configured: bool = False

        # Voice pipeline components
        self.voice_router: VoiceRouter | None = None
        self.voice_registry: VoicePresetRegistry | None = None
        self._synthesis_stream: Any = None
        self.digest_processor: Any = None

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
            self.effects_library = None
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
            voice_backend=self._voice_adapter,
        )
        self.effects_library = EffectsPresetLibrary()
        if audio_config.creature_voice_presets:
            preset_data = {
                name: {"effects": preset.effects}
                for name, preset in audio_config.creature_voice_presets.items()
            }
            self.effects_library.load_from_dict(preset_data)

    def init_voice(self) -> None:
        """Initialize voice pipeline (router, TTS engine, digest processor)."""
        if self._voice_adapter is None:
            log.warning(
                "VOICE: pipeline disabled — no voice adapter (TTS will be text-only)"
            )
            self.voice_router = None
            self.voice_registry = None
            self._synthesis_stream = None
            self.digest_processor = None
            return

        # Load voice presets from genre pack if available
        voice_config: dict = {}
        if self._genre_pack is not None:
            pack_presets = getattr(self._genre_pack, "voice_presets", None)
            if isinstance(pack_presets, dict) and pack_presets:
                voice_config = pack_presets
            elif pack_presets is not None and hasattr(pack_presets, "model_dump"):
                voice_config = pack_presets.model_dump()
            else:
                _pack_dir = getattr(self._genre_pack, "pack_dir", None)
                if _pack_dir is not None:
                    candidate = Path(_pack_dir) / "voice_presets.yaml"
                    if candidate.exists():
                        import yaml

                        with open(candidate) as f:
                            voice_config = yaml.safe_load(f) or {}

        if voice_config:
            self.voice_registry = VoicePresetRegistry.from_genre_config(voice_config)
        else:
            self.voice_registry = VoicePresetRegistry()

        self.voice_router = VoiceRouter(registry=self.voice_registry)

        # TTS engine: Kokoro preferred, Piper fallback
        if self._enable_tts:
            try:
                from sidequest_daemon.voice.kokoro import KokoroEngine
                from sidequest_daemon.voice.model_manager import KokoroModelManager
                from sidequest_daemon.voice.stream import SynthesisStream

                model_manager = KokoroModelManager()
                engine = KokoroEngine(model_manager=model_manager)
                self._synthesis_stream = SynthesisStream(
                    engine, effects_library=self.effects_library
                )
            except Exception as exc:
                log.warning("Kokoro TTS unavailable, falling back to Piper: %s", exc)
                try:
                    from sidequest_daemon.voice.piper import PiperEngine
                    from sidequest_daemon.voice.stream import SynthesisStream as _SynthStream

                    engine = PiperEngine()
                    self._synthesis_stream = _SynthStream(
                        engine, effects_library=self.effects_library
                    )
                except Exception as piper_exc:
                    log.error(
                        "VOICE: both Kokoro and Piper TTS failed — voice narration unavailable: %s",
                        piper_exc,
                    )
                    self._synthesis_stream = None
        else:
            self._synthesis_stream = None

        # Validate required voice models from genre pack
        if (
            self._genre_pack is not None
            and self._genre_pack.required_voice_models
        ):
            from sidequest_daemon.voice.validation import validate_required_voice_models

            available_engines = []
            if self._synthesis_stream is not None:
                available_engines = [self._synthesis_stream._engine]
            validate_required_voice_models(self._genre_pack, available_engines)

        # Digest processor for long narrations
        from sidequest_daemon.voice.digest import DigestProcessor

        self.digest_processor = DigestProcessor()
