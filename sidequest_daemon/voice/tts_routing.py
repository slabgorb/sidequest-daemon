"""TTS voice routing — story 4-6.

Maps character/NPC IDs to voice presets from genre pack YAML config.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class TtsModel(Enum):
    """TTS engine model — typed, not raw strings."""

    Kokoro = "kokoro"
    Piper = "piper"


class AssignmentSource(Enum):
    """Tracks where a voice assignment came from."""

    GenrePackExplicit = "genre_pack_explicit"
    GenrePackDefault = "genre_pack_default"
    SessionOverride = "session_override"


class Speaker:
    """Identifies who is speaking in narration text."""

    def __init__(self, *, is_narrator: bool, character_id: Optional[str] = None) -> None:
        self.is_narrator = is_narrator
        self.character_id = character_id

    @classmethod
    def narrator(cls) -> Speaker:
        return cls(is_narrator=True)

    @classmethod
    def character(cls, character_id: str) -> Speaker:
        return cls(is_narrator=False, character_id=character_id)


class VoicePreset:
    """Voice preset with model, voice_id, and speed."""

    def __init__(self, *, model: TtsModel, voice_id: str, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.model = model
        self.voice_id = voice_id
        self.speed = speed


class VoiceAssignment:
    """Result of routing a speaker to a voice preset."""

    def __init__(
        self,
        *,
        character_id: str,
        preset: VoicePreset,
        source: AssignmentSource,
    ) -> None:
        self.character_id = character_id
        self.preset = preset
        self.source = source


class TtsVoiceRouter:
    """Routes character/NPC IDs to voice presets from genre pack config."""

    def __init__(
        self,
        *,
        narrator_preset: VoicePreset,
        default_npc_preset: VoicePreset,
        character_presets: dict[str, VoicePreset],
    ) -> None:
        self._narrator_preset = narrator_preset
        self._default_npc_preset = default_npc_preset
        self._character_presets = character_presets

    @classmethod
    def from_genre_pack(cls, config: dict) -> TtsVoiceRouter:
        """Parse a genre pack media config into a router."""
        vp = config["voice_presets"]

        narrator_preset = _parse_preset(vp["narrator"])
        default_npc_preset = _parse_preset(vp["default_npc"])

        character_presets: dict[str, VoicePreset] = {}
        for char_id, char_cfg in vp.get("characters", {}).items():
            character_presets[char_id] = _parse_preset(char_cfg)

        return cls(
            narrator_preset=narrator_preset,
            default_npc_preset=default_npc_preset,
            character_presets=character_presets,
        )

    def route(self, speaker: Speaker) -> VoiceAssignment:
        """Route a speaker to a voice assignment."""
        if speaker.is_narrator:
            return VoiceAssignment(
                character_id="narrator",
                preset=self._narrator_preset,
                source=AssignmentSource.GenrePackExplicit,
            )

        char_id = speaker.character_id or "unknown"

        if char_id in self._character_presets:
            return VoiceAssignment(
                character_id=char_id,
                preset=self._character_presets[char_id],
                source=AssignmentSource.GenrePackExplicit,
            )

        return VoiceAssignment(
            character_id=char_id,
            preset=self._default_npc_preset,
            source=AssignmentSource.GenrePackDefault,
        )


def identify_speaker(text: str, known_npcs: list[str]) -> Speaker:
    """Identify speaker from narration text patterns."""
    for npc in known_npcs:
        # Match "Name says:" or "Name:" at start of text
        if re.match(rf"^{re.escape(npc)}\s+says\s*:", text) or re.match(
            rf"^{re.escape(npc)}\s*:", text
        ):
            return Speaker.character(npc)
    return Speaker.narrator()


def _parse_preset(cfg: dict) -> VoicePreset:
    """Parse a voice preset dict from genre pack config."""
    return VoicePreset(
        model=TtsModel(cfg["model"]),
        voice_id=cfg["voice"],
        speed=float(cfg["speed"]),
    )
