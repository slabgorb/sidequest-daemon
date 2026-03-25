"""CharacterVoiceRouter — maps characters to Kokoro voice IDs.

Priority chain: explicit assignment > archetype > role > deterministic default.
"""

from __future__ import annotations

import hashlib

from sidequest_daemon.voice.kokoro import KOKORO_VOICES

_NUM_VOICES = len(KOKORO_VOICES)


class CharacterVoiceRouter:
    """Routes character/NPC identifiers to Kokoro voice IDs (0-53).

    Supports three layers of mapping with a deterministic fallback:
    1. Explicit per-character assignment via assign()
    2. Archetype-based mapping via load_archetype_map()
    3. Role-based deterministic hash
    4. Default deterministic hash from character_id
    """

    def __init__(self) -> None:
        self._explicit: dict[str, int] = {}
        self._archetype_map: dict[str, int] = {}

    def assign(self, character_id: str, voice_id: int) -> None:
        """Explicitly assign a voice_id to a character."""
        self._validate_voice_id(voice_id)
        self._explicit[character_id] = voice_id

    def load_archetype_map(self, mapping: dict[str, int]) -> None:
        """Load archetype→voice_id mappings. Validates all voice IDs."""
        for archetype, voice_id in mapping.items():
            self._validate_voice_id(voice_id)
        self._archetype_map.update(mapping)

    def get_voice_id(
        self,
        character_id: str,
        *,
        archetype: str | None = None,
        role: str | None = None,
    ) -> int:
        """Resolve a voice_id for a character.

        Priority: explicit > archetype > role hash > character_id hash.
        """
        # 1. Explicit assignment
        if character_id in self._explicit:
            return self._explicit[character_id]

        # 2. Archetype mapping
        if archetype and archetype in self._archetype_map:
            return self._archetype_map[archetype]

        # 3. Role-based deterministic hash
        if role:
            return self._hash_to_voice_id(role)

        # 4. Default: deterministic hash from character_id
        return self._hash_to_voice_id(character_id)

    @staticmethod
    def _validate_voice_id(voice_id: int) -> None:
        if voice_id < 0 or voice_id >= _NUM_VOICES:
            raise ValueError(
                f"voice_id must be 0-{_NUM_VOICES - 1}, got {voice_id}"
            )

    @staticmethod
    def _hash_to_voice_id(key: str) -> int:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % _NUM_VOICES
