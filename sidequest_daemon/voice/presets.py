"""Effects preset library and character-to-preset routing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Built-in presets available without any genre pack.
_BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "dry": {"effects": []},
    "bright": {
        "effects": [
            {"type": "highpass_filter", "params": {"cutoff_hz": 200}},
            {"type": "gain", "params": {"gain_db": 2}},
        ],
    },
    "dark": {
        "effects": [
            {"type": "lowpass_filter", "params": {"cutoff_hz": 4000}},
            {"type": "gain", "params": {"gain_db": -1}},
        ],
    },
    "intimate": {
        "effects": [
            {"type": "compressor", "params": {"threshold_db": -18}},
            {"type": "reverb", "params": {"room_size": 0.25}},
        ],
    },
}


class EffectsPresetLibrary:
    """Manages named effects presets with built-in defaults and genre-pack loading."""

    def __init__(self) -> None:
        self._presets: dict[str, dict[str, Any]] = deepcopy(_BUILTIN_PRESETS)

    def get_preset(self, name: str) -> dict[str, Any]:
        """Return the preset config for *name*, or raise ``KeyError``."""
        if name not in self._presets:
            raise KeyError(f"Unknown preset: {name}")
        return deepcopy(self._presets[name])

    def list_presets(self) -> list[str]:
        """Return sorted list of available preset names."""
        return sorted(self._presets)

    def load_from_dict(self, data: dict[str, dict[str, Any]]) -> None:
        """Load (or override) presets from a YAML-style dictionary."""
        for name, config in data.items():
            self._presets[name] = deepcopy(config)

    def compose_presets(self, names: list[str]) -> dict[str, Any]:
        """Merge multiple presets into one by concatenating their effects."""
        if not names:
            return {"effects": []}
        if len(names) == 1:
            return self.get_preset(names[0])
        combined: list[dict[str, Any]] = []
        for name in names:
            preset = self.get_preset(name)
            combined.extend(preset.get("effects", []))
        return {"effects": combined}


class CharacterEffectsRouter:
    """Maps character IDs to effect presets."""

    def __init__(
        self,
        library: EffectsPresetLibrary,
        default_preset: str = "dry",
    ) -> None:
        self.library = library
        self.default_preset = default_preset
        self._mapping: dict[str, str] = {}

    def map_character(self, character_id: str, preset_name: str) -> None:
        """Assign *preset_name* to *character_id*. Raises ``KeyError`` if preset unknown."""
        # Validate the preset exists.
        self.library.get_preset(preset_name)
        self._mapping[character_id] = preset_name

    def get_preset_for_character(self, character_id: str) -> dict[str, Any]:
        """Return the preset config for *character_id*, falling back to default."""
        preset_name = self._mapping.get(character_id, self.default_preset)
        return self.library.get_preset(preset_name)
