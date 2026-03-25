"""NPC voice assignment — assigns unique Kokoro voice IDs to NPCs."""

from __future__ import annotations

import hashlib

from sidequest_daemon.types import HasVoice
from sidequest_daemon.voice.kokoro import KOKORO_VOICES
from sidequest_daemon.voice.protocol import VoicePreset
from sidequest_daemon.voice.registry import VoicePresetRegistry


def assign_voices(npcs: list[HasVoice]) -> None:
    """Assign a unique Kokoro voice_id to each NPC that lacks one.

    - Preserves existing voice_id assignments.
    - Deterministic: same NPC names produce the same voice mapping.
    - When more NPCs than voices (>54), wraps with modulo.
    """
    if not npcs:
        return

    num_voices = len(KOKORO_VOICES)
    reserved = {npc.voice_id for npc in npcs if npc.voice_id is not None}

    for npc in npcs:
        if npc.voice_id is not None:
            continue

        # Deterministic hash from NPC name
        digest = hashlib.sha256(npc.name.encode("utf-8")).digest()
        base = int.from_bytes(digest[:4], "big") % num_voices

        # Find an unused slot (skip reserved IDs)
        voice_id = base
        attempts = 0
        while voice_id in reserved and attempts < num_voices:
            voice_id = (voice_id + 1) % num_voices
            attempts += 1

        npc.voice_id = voice_id
        reserved.add(voice_id)


def register_npc_voices(npcs: list[HasVoice], registry: VoicePresetRegistry) -> None:
    """Populate a VoicePresetRegistry with presets derived from NPC voice_ids.

    Skips NPCs without a voice_id and does not overwrite existing entries.
    """
    for npc in npcs:
        if npc.voice_id is None:
            continue

        # Don't overwrite existing registry entries
        existing = registry.list_characters()
        if npc.name in existing:
            continue

        preset = VoicePreset(
            name=f"{npc.name.lower()}_voice",
            model=str(npc.voice_id),
        )
        registry.register(npc.name, preset)
