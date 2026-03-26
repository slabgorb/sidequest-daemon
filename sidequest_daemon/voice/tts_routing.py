"""TTS voice routing — stubs for story 4-6.

Maps character/NPC IDs to voice presets from genre pack YAML config.
Implementation intentionally empty — RED phase stubs only.
"""

from __future__ import annotations

from enum import Enum


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

    pass


class VoicePreset:
    """Voice preset with model, voice_id, and speed."""

    pass


class VoiceAssignment:
    """Result of routing a speaker to a voice preset."""

    pass


class TtsVoiceRouter:
    """Routes character/NPC IDs to voice presets from genre pack config."""

    pass


def identify_speaker(text: str, known_npcs: list[str]) -> Speaker:
    """Identify speaker from narration text patterns."""
    raise NotImplementedError
