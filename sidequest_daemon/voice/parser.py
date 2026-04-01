"""Narrative segment parser — splits narrative text into speaker segments.

Identifies narrator vs character speech by detecting dialogue tags
(e.g., 'Aldric said, "..."' or '"..." said Aldric').
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class NarrativeSegment(BaseModel):
    """A segment of narrative text attributed to a speaker.

    speaker is None for narration, a character name string for dialogue.
    """

    text: str
    speaker: str | None = None

    @property
    def is_narration(self) -> bool:
        return self.speaker is None


# Speech verbs used to detect dialogue tags
_SPEECH_VERBS = (
    r"said|says|whispered|shouted|called|cried|asked|replied|answered|"
    r"exclaimed|muttered|murmured|demanded|insisted|warned|commanded|"
    r"yelled|screamed|declared|announced|stated|suggested|pleaded|"
    r"growled|hissed|snarled|gasped|stammered|stuttered|bellowed"
)

# Pattern: "dialogue" Name verb (e.g. "Watch out," Aldric warned.)
_QUOTE_NAME_VERB = re.compile(
    rf"""
    ["\u201c](?P<speech>[^"\u201d]+)["\u201d]      # quoted speech
    [,.]?\s*                                         # optional comma/period + space
    (?:the\s+)?                                      # optional "the"
    (?P<name>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)  # capitalized name
    \s+(?:{_SPEECH_VERBS})                           # speech verb
    """,
    re.VERBOSE,
)

# Pattern: "dialogue" verb Name (e.g. "I will go," said Aldric.)
_QUOTE_VERB_NAME = re.compile(
    rf"""
    ["\u201c](?P<speech>[^"\u201d]+)["\u201d]      # quoted speech
    [,.]?\s*                                         # optional comma/period + space
    (?:{_SPEECH_VERBS})\s+                           # speech verb
    (?P<name>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)  # capitalized name
    """,
    re.VERBOSE,
)

# Pattern: Name verb, "dialogue"
_TAG_THEN_QUOTE = re.compile(
    rf"""
    (?P<name>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)  # capitalized name
    \s+(?:{_SPEECH_VERBS})                            # speech verb
    [,:]?\s*                                          # optional comma/colon + space
    ["\u201c](?P<speech>[^"\u201d]+)["\u201d]         # quoted speech
    """,
    re.VERBOSE,
)

# Pattern: "dialogue" the <name> verb (e.g. "Halt!" the guard demanded.)
_QUOTE_THE_NAME_VERB = re.compile(
    rf"""
    ["\u201c](?P<speech>[^"\u201d]+)["\u201d]      # quoted speech
    [,.]?\s*                                         # optional comma/period + space
    the\s+                                           # "the"
    (?P<name>[a-zA-Z]+(?:\s+[a-zA-Z]+)*)            # name (may be lowercase)
    \s+(?:{_SPEECH_VERBS})                           # speech verb
    """,
    re.VERBOSE,
)


class NarrativeSegmentParser:
    """Parses narrative text into a list of NarrativeSegments."""

    def __init__(self, known_npcs: list[str] | None = None) -> None:
        self._known_npcs: list[str] = known_npcs or []

    def set_known_npcs(self, npcs: list[str]) -> None:
        """Update the known NPC list for speaker identification."""
        self._known_npcs = list(npcs)

    def parse(self, text: str) -> list[NarrativeSegment]:
        if not text or not text.strip():
            return []

        # Pre-check: use identify_speaker for "Name says:" / "Name:" patterns
        # against known NPCs. This handles dialogue attribution that the regex
        # patterns below may miss (e.g., unquoted dialogue after a colon).
        if self._known_npcs:
            from sidequest_daemon.voice.tts_routing import identify_speaker

            speaker = identify_speaker(text, self._known_npcs)
            if not speaker.is_narrator and speaker.character_id:
                # The entire text is attributed to this speaker
                # Strip the "Name says:" or "Name:" prefix
                for npc in self._known_npcs:
                    prefix_patterns = [f"{npc} says:", f"{npc}:"]
                    for prefix in prefix_patterns:
                        if text.strip().startswith(prefix):
                            speech = text.strip()[len(prefix):].strip()
                            if speech:
                                return [NarrativeSegment(text=speech, speaker=speaker.character_id)]

        segments: list[NarrativeSegment] = []
        # Collect all dialogue matches with their spans
        matches: list[tuple[int, int, str, str]] = []  # (start, end, name, speech)

        for pattern in (
            _QUOTE_NAME_VERB,
            _QUOTE_VERB_NAME,
            _TAG_THEN_QUOTE,
            _QUOTE_THE_NAME_VERB,
        ):
            for m in pattern.finditer(text):
                name = m.group("name")
                # Title-case names from "the guard" patterns
                name = name.title()
                matches.append((m.start(), m.end(), name, m.group("speech")))

        if not matches:
            # No dialogue found — everything is narration
            # Check if text is just a bare quote with no tag
            return [NarrativeSegment(text=text, speaker=None)]

        # Sort by position
        matches.sort(key=lambda x: x[0])

        # Deduplicate overlapping matches (keep first)
        deduped: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, name, speech in matches:
            if start >= last_end:
                deduped.append((start, end, name, speech))
                last_end = end

        pos = 0
        for start, end, name, speech in deduped:
            # Narration before this dialogue
            if start > pos:
                narration = text[pos:start].strip()
                if narration:
                    segments.append(NarrativeSegment(text=narration, speaker=None))

            # The dialogue segment
            segments.append(NarrativeSegment(text=speech, speaker=name))
            pos = end

        # Trailing narration
        if pos < len(text):
            trailing = text[pos:].strip()
            if trailing:
                segments.append(NarrativeSegment(text=trailing, speaker=None))

        return segments
