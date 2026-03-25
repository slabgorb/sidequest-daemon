"""Post-processing effects chain wrapping Pedalboard's Pedalboard API."""

from __future__ import annotations

from typing import Any

import numpy as np

import pedalboard  # type: ignore[import-untyped]


# Map serialized type names to pedalboard effect classes.
EFFECT_TYPE_MAP: dict[str, type] = {
    "reverb": pedalboard.Reverb,
    "compressor": pedalboard.Compressor,
    "highpass_filter": pedalboard.HighpassFilter,
    "lowpass_filter": pedalboard.LowpassFilter,
    "gain": pedalboard.Gain,
    "pitch_shift": pedalboard.PitchShift,
}


# Map shorthand param names to pedalboard's actual parameter names.
_PARAM_ALIASES: dict[str, str] = {
    "cutoff_hz": "cutoff_frequency_hz",
}


class PedalboardEffectsChain:
    """Wrapper around pedalboard.Pedalboard for processing audio through effects."""

    def __init__(self, effects: list[Any] | None = None) -> None:
        self.effects: list[Any] = effects if effects is not None else []
        self.board = pedalboard.Pedalboard(self.effects)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        """Process *audio* through the effects chain.

        Accepts 1-D (mono) or 2-D (channels, samples) float32 arrays.
        """
        was_1d = audio.ndim == 1
        if was_1d:
            audio = audio[np.newaxis, :]

        if len(self.effects) == 0:
            result = audio
        else:
            result = self.board(audio, sample_rate)
            if not isinstance(result, np.ndarray):
                try:
                    arr = np.asarray(result, dtype=np.float32)
                    # If conversion produced a usable array, keep it;
                    # otherwise fall back to the original audio.
                    result = arr if arr.size > 0 else audio
                except (TypeError, ValueError):
                    result = audio

        # Ensure output matches input shape.
        if result.shape != audio.shape:
            try:
                result = result.reshape(audio.shape)
            except ValueError:
                result = audio

        result = result.astype(np.float32, copy=False)

        if was_1d:
            result = result.squeeze(0)
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chain to a dictionary."""
        serialized: list[dict[str, Any]] = []
        for effect in self.effects:
            for type_name, cls in EFFECT_TYPE_MAP.items():
                if isinstance(effect, cls):
                    serialized.append(
                        {
                            "type": type_name,
                            "params": getattr(effect, "_params", {}),
                        }
                    )
                    break
            else:
                serialized.append(
                    {
                        "type": type(effect).__name__.lower(),
                        "params": getattr(effect, "_params", {}),
                    }
                )
        return {"effects": serialized}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PedalboardEffectsChain:
        """Reconstruct a chain from a serialized dictionary."""
        effects: list[Any] = []
        for entry in config.get("effects", []):
            type_name = entry["type"]
            params = entry.get("params", {})
            if type_name not in EFFECT_TYPE_MAP:
                raise ValueError(
                    f"Unknown effect type: {type_name}. "
                    f"Supported: {', '.join(EFFECT_TYPE_MAP)}"
                )
            resolved = {_PARAM_ALIASES.get(k, k): v for k, v in params.items()}
            effect = EFFECT_TYPE_MAP[type_name](**resolved)
            try:
                effect._params = params
            except AttributeError:
                pass  # pedalboard native objects use __slots__
            effects.append(effect)
        return cls(effects=effects)


class EffectsProcessor:
    """High-level processor that routes audio through character-specific effects."""

    def __init__(
        self,
        preset_library: Any,
        router: Any | None = None,
        default_preset: str = "dry",
    ) -> None:
        self.preset_library = preset_library
        self.router = router
        self.default_preset = default_preset

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        character_id: str | None = None,
        preset_name: str | None = None,
        dry_run: bool = False,
    ) -> np.ndarray:
        """Process audio through the appropriate effects chain."""
        if dry_run:
            return audio

        # Determine which preset to use.
        if preset_name is not None:
            preset_config = self.preset_library.get_preset(preset_name)
        elif character_id is not None and self.router is not None:
            preset_config = self.router.get_preset_for_character(character_id)
        elif character_id is not None:
            try:
                preset_config = self.preset_library.get_preset(character_id)
            except KeyError:
                preset_config = self.preset_library.get_preset(self.default_preset)
        else:
            preset_config = self.preset_library.get_preset(self.default_preset)

        chain = PedalboardEffectsChain.from_dict(preset_config)
        return chain.process(audio, sample_rate=sample_rate)
