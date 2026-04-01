"""Voice router — routes narrative text to (text, VoicePreset) pairs."""

from __future__ import annotations

from sidequest_daemon.voice.parser import NarrativeSegmentParser
from sidequest_daemon.voice.protocol import VoicePreset
from sidequest_daemon.voice.registry import VoicePresetRegistry


class VoiceRouter:
    """Routes narrative text through parser and registry to produce voiced segments."""

    def __init__(self, registry: VoicePresetRegistry | None = None) -> None:
        self.registry = registry or VoicePresetRegistry()
        # Pass known character names to parser for identify_speaker wiring
        known_npcs = self.registry.list_characters()
        self.parser = NarrativeSegmentParser(known_npcs=known_npcs)

    def update_known_npcs(self, npcs: list[str]) -> None:
        """Update the parser's known NPC list (call after registering new voices)."""
        self.parser.set_known_npcs(npcs)

    def route(self, text: str) -> list[tuple[str, VoicePreset]]:
        segments = self.parser.parse(text)
        pairs: list[tuple[str, VoicePreset]] = []

        for segment in segments:
            if segment.is_narration:
                preset = self.registry.get_narrator_preset()
            else:
                preset = self.registry.get(segment.speaker)
            pairs.append((segment.text, preset))

        return pairs
