"""TTS speaker identification — dialogue attribution for voice routing.

Identifies who is speaking in narration text based on known NPC names.
Used by parser.py for NPC dialogue attribution to Kokoro voice synthesis.
"""

from __future__ import annotations

import re
from typing import Optional


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


def identify_speaker(text: str, known_npcs: list[str]) -> Speaker:
    """Identify speaker from narration text patterns."""
    for npc in known_npcs:
        # Match "Name says:" or "Name:" at start of text
        if re.match(rf"^{re.escape(npc)}\s+says\s*:", text) or re.match(
            rf"^{re.escape(npc)}\s*:", text
        ):
            return Speaker.character(npc)
    return Speaker.narrator()
