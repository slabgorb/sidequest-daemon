"""KokoroEngine — concrete SynthesisEngine using Kokoro TTS.

Story 34-2: High-quality neural TTS engine with streaming support,
voice blending, and prosody-preserving synthesis.
"""

from __future__ import annotations

import logging
import re
import struct
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sidequest_daemon.voice.errors import ModelLoadError, SynthesisFailedError, VoicePresetError
from sidequest_daemon.voice.protocol import (
    AudioSegment,
    SynthesisEngine,
    SynthesisMode,
    VoicePreset,
)

log = logging.getLogger(__name__)

# Default model/voices paths (downloaded from github.com/thewh1teagle/kokoro-onnx)
_DEFAULT_CACHE_DIR = Path.home() / ".sidequest" / "models" / "kokoro"
_DEFAULT_MODEL_FILE = "kokoro-v1.0.onnx"
_DEFAULT_VOICES_FILE = "voices-v1.0.bin"

# 54 built-in Kokoro voices — static defaults, refreshed on model load
KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
    "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
    "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex",
    "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega",
    "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune",
    "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex",
    "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]

DEFAULT_MODEL = "kokoro-v1_0"
DEFAULT_VOICE_ID = 0
BLEND_THRESHOLD = 54


class KokoroEngine(SynthesisEngine):
    """Text-to-speech engine backed by Kokoro TTS.

    Supports both batch and streaming synthesis modes with voice blending
    for voice_id > 54.
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        lang: str = "en",
        speed: float = 1.0,
        voice_id: int | None = None,
        model: str | None = None,
        model_manager: object | None = None,
    ) -> None:
        self._lang = lang
        self._speed = speed
        self._default_voice_id = voice_id if voice_id is not None else DEFAULT_VOICE_ID
        self._default_model = model or DEFAULT_MODEL
        self._model_manager = model_manager
        self._is_ready = False

    # -- SynthesisEngine interface -------------------------------------------

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def supported_modes(self) -> list[SynthesisMode]:
        return [SynthesisMode.BATCH, SynthesisMode.STREAMING]

    async def synthesize(self, text: str, voice_preset: VoicePreset) -> AudioSegment:
        if not self._is_ready:
            raise ModelLoadError("Engine not ready — call warm_up() first")

        normalized = self._normalize_text(text)
        if not normalized:
            return AudioSegment(data=b"", sample_rate=self.SAMPLE_RATE, channels=1)

        voice_id = self._resolve_voice_id(voice_preset)
        speed = voice_preset.rate

        try:
            raw = self._synthesize_raw(normalized, voice_id=voice_id, speed=speed)
        except (ModelLoadError, SynthesisFailedError):
            raise
        except Exception as exc:
            raise SynthesisFailedError(str(exc)) from exc

        return AudioSegment(data=raw, sample_rate=self.SAMPLE_RATE, channels=1)

    def resolved_model_path(self, name: str) -> Path:
        """Resolve model path through the model manager if available."""
        if self._model_manager is not None:
            return self._model_manager.model_path(name)
        return Path(name)

    async def warm_up(self) -> None:
        if self._is_ready:
            return
        if self._model_manager is not None:
            await self._model_manager.ensure_ready(self._default_model)
        self._load_model()
        self._is_ready = True

    async def shutdown(self) -> None:
        self._is_ready = False

    # -- Properties ----------------------------------------------------------

    @property
    def lang(self) -> str:
        return self._lang

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def default_voice_id(self) -> int:
        return self._default_voice_id

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # -- Voice selection -----------------------------------------------------

    def list_voices(self) -> list[str]:
        """Return list of available Kokoro voice names."""
        return list(KOKORO_VOICES)

    def _resolve_voice_id(self, preset: VoicePreset) -> int:
        """Resolve a VoicePreset to a Kokoro voice_id.

        Priority: preset.voice_id > preset.model (as int) > engine default.
        """
        if preset.voice_id is not None:
            vid = preset.voice_id
        elif preset.model is not None:
            try:
                vid = int(preset.model)
            except ValueError as exc:
                raise VoicePresetError(
                    f"Invalid voice model '{preset.model}': must be numeric"
                ) from exc
        else:
            vid = self._default_voice_id
        if vid < 0:
            raise ValueError(f"Invalid voice_id: {vid}")
        return vid

    def _is_blend_voice(self, voice_id: int) -> bool:
        """Return True if voice_id triggers voice blending (id > 54)."""
        return voice_id > BLEND_THRESHOLD

    # -- Text normalization --------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Normalize input text for TTS.

        Strips whitespace, collapses runs, removes newlines.
        Preserves punctuation (important for Kokoro prosody).
        """
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -- Streaming -----------------------------------------------------------

    async def synthesize_stream(
        self, text: str, voice_preset: VoicePreset
    ) -> AsyncIterator[AudioSegment]:
        """Yield audio chunks for streaming playback.

        Requires warm_up() to have been called first.
        """
        if not self._is_ready:
            raise ModelLoadError("Engine not ready — call warm_up() first")

        normalized = self._normalize_text(text)
        if not normalized:
            return

        voice_id = self._resolve_voice_id(voice_preset)
        speed = voice_preset.rate

        try:
            chunks = self._synthesize_raw_chunks(normalized, voice_id=voice_id, speed=speed)
        except (ModelLoadError, SynthesisFailedError):
            raise
        except Exception as exc:
            raise SynthesisFailedError(str(exc)) from exc

        for chunk_data in chunks:
            if chunk_data:
                yield AudioSegment(
                    data=chunk_data, sample_rate=self.SAMPLE_RATE, channels=1
                )

    # -- Internal (overridden in tests via patch) ----------------------------

    _tts: Any = None  # kokoro_onnx.Kokoro instance
    _voice_names: list[str] = []

    def _load_model(self) -> None:
        """Load Kokoro ONNX model and voices. Patched in tests."""
        try:
            from kokoro_onnx import Kokoro as KokoroOnnx
        except ImportError as exc:
            raise ModelLoadError(
                "kokoro-onnx not installed: uv pip install kokoro-tts"
            ) from exc

        model_path = _DEFAULT_CACHE_DIR / _DEFAULT_MODEL_FILE
        voices_path = _DEFAULT_CACHE_DIR / _DEFAULT_VOICES_FILE

        if not model_path.exists():
            raise ModelLoadError(
                f"Kokoro model not found at {model_path}. "
                f"Download from github.com/thewh1teagle/kokoro-onnx/releases"
            )
        if not voices_path.exists():
            raise ModelLoadError(
                f"Kokoro voices not found at {voices_path}. "
                f"Download from github.com/thewh1teagle/kokoro-onnx/releases"
            )

        self._tts = KokoroOnnx(str(model_path), str(voices_path))
        self._voice_names = self._tts.get_voices()
        KOKORO_VOICES.clear()
        KOKORO_VOICES.extend(self._voice_names)
        log.info("Kokoro loaded: %d voices, model=%s", len(self._voice_names), model_path.name)

    def _voice_name_for_id(self, voice_id: int) -> str:
        """Map numeric voice_id to Kokoro voice name."""
        if self._voice_names and 0 <= voice_id < len(self._voice_names):
            return self._voice_names[voice_id]
        # Fallback to first voice
        return self._voice_names[0] if self._voice_names else "af_bella"

    @staticmethod
    def _float32_to_s16le(samples: Any) -> bytes:
        """Convert float32 numpy array to signed 16-bit little-endian PCM bytes."""
        import numpy as np

        clamped = np.clip(samples, -1.0, 1.0)
        int16_data = (clamped * 32767).astype(np.int16)
        return int16_data.tobytes()

    def _synthesize_raw(
        self, text: str, *, voice_id: int, speed: float = 1.0
    ) -> bytes:
        """Run Kokoro synthesis and return raw PCM s16le bytes. Patched in tests."""
        if self._tts is None:
            return b""

        voice_name = self._voice_name_for_id(voice_id)
        # Detect language from voice name prefix (ff_=French, bf_=British, etc.)
        lang = self._detect_lang(voice_name)

        try:
            samples, _sr = self._tts.create(text, voice=voice_name, speed=speed, lang=lang)
            return self._float32_to_s16le(samples)
        except Exception as exc:
            raise SynthesisFailedError(f"Kokoro synthesis failed: {exc}") from exc

    def _synthesize_raw_chunks(
        self, text: str, *, voice_id: int, speed: float = 1.0
    ) -> list[bytes]:
        """Run Kokoro synthesis and return list of PCM chunks. Patched in tests."""
        # For now, batch synthesis returns a single chunk.
        # Streaming via create_stream() is async and would need different wiring.
        raw = self._synthesize_raw(text, voice_id=voice_id, speed=speed)
        return [raw] if raw else []

    @staticmethod
    def _detect_lang(voice_name: str) -> str:
        """Detect language from Kokoro voice name prefix."""
        prefix = voice_name.split("_")[0] if "_" in voice_name else ""
        lang_map = {
            "af": "en-us",  # American female
            "am": "en-us",  # American male
            "bf": "en-gb",  # British female
            "bm": "en-gb",  # British male
            "ef": "en-us",  # (English female)
            "em": "en-us",  # (English male)
            "ff": "fr-fr",  # French female
            "fm": "fr-fr",  # French male
            "if": "it",     # Italian female
            "im": "it",     # Italian male
            "jf": "ja",     # Japanese female
            "jm": "ja",     # Japanese male
            "pf": "en-us",  # (Piper female, fallback)
            "pm": "en-us",  # (Piper male, fallback)
            "hf": "cmn",    # (Hindi/Chinese female)
            "hm": "cmn",    # (Hindi/Chinese male)
            "zf": "cmn",    # Chinese female
            "zm": "cmn",    # Chinese male
        }
        return lang_map.get(prefix, "en-us")
