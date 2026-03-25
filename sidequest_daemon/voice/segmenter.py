"""Sentence segmenter — split narrative text into sentence-level chunks for synthesis."""

from __future__ import annotations

import re


# Abbreviations that should not trigger a sentence split.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "etc",
        "vs",
        "vol",
        "dept",
        "est",
        "approx",
        "inc",
        "ltd",
        "gen",
        "gov",
        "sgt",
        "cpl",
        "pvt",
        "capt",
        "lt",
        "col",
    }
)

# Sentence-boundary patterns.  Order matters — earlier alternatives take priority.
_SPLIT_RE = re.compile(
    r"""
    # 1. Ellipsis followed by a new sentence (capital letter or opening quote)
    (?:\.{3}|\u2026)
    (?=\s+[A-Z"\u201c])

    # 2. Period (or period + closing quote) as sentence terminator
    |(?<!\.)\.
    ["\u201d]?
    (?=\s|$)

    # 3. !?" followed by whitespace + opening quote  → two quoted sentences
    |[!?]["\u201d]
    (?=\s+["\u201c])

    # 4. Bare ! or ? (no closing quote) as sentence terminator
    |[!?]
    (?=\s|$)

    # 5. !?" at end-of-string
    |[!?]["\u201d]$
    """,
    re.VERBOSE,
)


class SentenceSegmenter:
    """Break narrative text into sentence-level semantic units."""

    def segment(self, text: str) -> list[str]:
        """Split *text* into sentences, preserving punctuation.

        Returns a list of stripped sentence strings.  Empty input yields an
        empty list.
        """
        if not text or not text.strip():
            return []

        sentences: list[str] = []
        last = 0

        for match in _SPLIT_RE.finditer(text):
            end = match.end()
            candidate = text[last:end].strip()

            # Check if the period belongs to an abbreviation.
            if match.group().startswith(".") and not match.group().startswith("..."):
                word_before = self._word_before_dot(text, match.start())
                if word_before and word_before.lower() in _ABBREVIATIONS:
                    continue

            if candidate:
                sentences.append(candidate)
            last = end

        # Remainder after the last split point.
        remainder = text[last:].strip()
        if remainder:
            sentences.append(remainder)

        return sentences

    @staticmethod
    def _word_before_dot(text: str, dot_pos: int) -> str | None:
        """Return the word immediately before the dot at *dot_pos*."""
        i = dot_pos - 1
        while i >= 0 and text[i].isalpha():
            i -= 1
        word = text[i + 1 : dot_pos]
        return word if word else None
