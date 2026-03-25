"""Genre pack voice configuration — YAML voice assignments per role."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from sidequest_daemon.voice.protocol import VoicePreset


class VoiceRoleAssignment(BaseModel):
    """Voice configuration for a single role (narrator, npc, combat)."""

    engine: Literal["piper", "kokoro"]
    voice_id: int | None = None
    model: str | None = None
    pitch: float = 1.0
    rate: float = 1.0
    effects: list[str | dict] = []

    def to_voice_preset(self, name: str) -> VoicePreset:
        """Convert to a VoicePreset."""
        return VoicePreset(
            name=name,
            pitch=self.pitch,
            rate=self.rate,
            effects=self.effects,
            model=self.model,
            voice_id=self.voice_id,
        )


_DEFAULT_NARRATOR = VoiceRoleAssignment(engine="piper")
_DEFAULT_NPC = VoiceRoleAssignment(engine="piper")


class GenreVoiceConfig(BaseModel):
    """Voice assignments per role, loadable from genre pack YAML."""

    narrator: VoiceRoleAssignment = _DEFAULT_NARRATOR
    npc_default: VoiceRoleAssignment = _DEFAULT_NPC
    combat: VoiceRoleAssignment | None = None
