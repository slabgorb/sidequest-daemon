"""Voice preset registry — maps characters to voice presets."""

from __future__ import annotations

from sidequest_daemon.voice.protocol import VoicePreset


_DEFAULT_NARRATOR = VoicePreset(name="narrator", pitch=1.0, rate=1.0)
_DEFAULT_FALLBACK = VoicePreset(name="fallback", pitch=1.0, rate=1.1)


class VoicePresetRegistry:
    """Maps character names to VoicePreset instances.

    Provides a narrator preset and a fallback for unknown characters.
    """

    def __init__(
        self,
        narrator_preset: VoicePreset | None = None,
        fallback_preset: VoicePreset | None = None,
    ) -> None:
        self._narrator = narrator_preset or _DEFAULT_NARRATOR
        self._fallback = fallback_preset or _DEFAULT_FALLBACK
        self._characters: dict[str, tuple[str, VoicePreset]] = {}

    def get_narrator_preset(self) -> VoicePreset:
        return self._narrator

    def register(self, character: str, preset: VoicePreset) -> None:
        self._characters[character.lower()] = (character, preset)

    def get(self, character: str) -> VoicePreset:
        entry = self._characters.get(character.lower())
        return entry[1] if entry else self._fallback

    def list_characters(self) -> list[str]:
        return [original for original, _ in self._characters.values()]

    @classmethod
    def _normalize_effects(cls, data: dict) -> dict:
        """Normalize effects from YAML format (list[dict]) to model format (list[str])."""
        effects = data.get("effects")
        if effects and isinstance(effects, list) and effects and isinstance(effects[0], dict):
            data = {**data, "effects": [e.get("type", "") for e in effects if isinstance(e, dict)]}
        return data

    @classmethod
    def from_genre_config(cls, config: dict) -> VoicePresetRegistry:
        narrator_data = config.get("narrator")
        narrator_preset = None
        if narrator_data:
            narrator_data = cls._normalize_effects(narrator_data)
            narrator_preset = VoicePreset(name="narrator", **narrator_data)

        registry = cls(narrator_preset=narrator_preset)

        characters = config.get("characters", {})
        for name, char_data in characters.items():
            char_data = cls._normalize_effects(char_data)
            preset = VoicePreset(name=name.lower() + "_voice", **char_data)
            registry.register(name, preset)

        return registry

    @classmethod
    def from_voice_config(cls, config) -> VoicePresetRegistry:
        """Build a registry from a GenreVoiceConfig."""
        narrator_preset = config.narrator.to_voice_preset("narrator")
        fallback_preset = config.npc_default.to_voice_preset("npc_default")
        return cls(narrator_preset=narrator_preset, fallback_preset=fallback_preset)
