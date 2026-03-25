"""Voice digest mode — condense long narration before TTS synthesis.

Story 13-7: When narration text exceeds a configurable character threshold,
summarize it via LLM before sending through the TTS pipeline. Full narration
is always delivered as text; digest mode only affects what gets spoken aloud.

Drama scoring hook: high-drama text gets a higher effective threshold so
climactic moments keep more spoken detail.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Drama keywords that boost the effective threshold (keep more text spoken)
_DRAMA_KEYWORDS = frozenset({
    "attack", "battle", "blood", "breathe", "clash", "collapse", "crumble",
    "crush", "death", "destroy", "devastating", "dodge", "dragon", "duel",
    "engulf", "escape", "explode", "explosion", "fall", "fight", "fire",
    "fireball", "flame", "flee", "fury", "howl", "inferno", "kill",
    "lightning", "rage", "roar", "scream", "shatter", "slash", "smash",
    "strike", "sword", "terror", "thunder", "unleash", "war", "wound",
})

_WORD_RE = re.compile(r"[a-z]+")


class DigestProcessor:
    """Condenses long narration text for TTS synthesis."""

    def __init__(
        self,
        *,
        threshold_chars: int = 800,
        drama_multiplier: float = 1.5,
    ) -> None:
        self.threshold_chars = threshold_chars
        self.drama_multiplier = drama_multiplier

    def should_digest(self, text: str, *, drama_weight: float | None = None) -> bool:
        """Return True if text exceeds the effective threshold."""
        return len(text) > self.effective_threshold(text, drama_weight=drama_weight)

    def effective_threshold(self, text: str, *, drama_weight: float | None = None) -> float:
        """Compute threshold adjusted by drama score or external drama_weight.

        When drama_weight is provided (from the combat pacing system), it
        overrides keyword-based scoring:
          - > 0.70: returns inf (no digest — full cinematic text spoken)
          - 0.30-0.70: uses base threshold (standard digest)
          - < 0.30: reduces threshold (aggressive digest for routine combat)

        When drama_weight is None, falls back to keyword-based drama_score.
        """
        if drama_weight is not None:
            # High drama: never digest — full text is the cinematic moment
            if drama_weight > 0.70:
                return float("inf")
            # Low drama: aggressive digest for routine/punchy combat
            if drama_weight < 0.30:
                # Scale from 0.25x at 0.0 to 1.0x at 0.30
                scale = 0.25 + 0.75 * (drama_weight / 0.30)
                return self.threshold_chars * scale
            # Mid drama: standard threshold
            return float(self.threshold_chars)

        # Fallback: keyword-based drama scoring
        score = self.drama_score(text)
        # Higher drama → higher threshold → more text kept
        multiplier = 1.0 + (self.drama_multiplier - 1.0) * score
        return self.threshold_chars * multiplier

    def drama_score(self, text: str) -> float:
        """Score text drama from 0.0 (idle) to 1.0 (climactic).

        Uses keyword density + punctuation intensity as heuristics.
        This is the hook for future playtesting tuning.
        """
        lower = text.lower()
        words = _WORD_RE.findall(lower)
        if not words:
            return 0.0

        # Keyword density
        drama_count = sum(1 for w in words if w in _DRAMA_KEYWORDS)
        keyword_ratio = min(drama_count / max(len(words), 1), 1.0)

        # Punctuation intensity (! and CAPS)
        excl_ratio = min(text.count("!") / max(len(text) / 100, 1), 1.0)
        caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        caps_ratio = min(caps_words / max(len(words), 1), 1.0)

        # Weighted combination
        score = keyword_ratio * 0.6 + excl_ratio * 0.25 + caps_ratio * 0.15
        return min(max(score, 0.0), 1.0)

    async def digest(self, text: str) -> str:
        """Summarize text for TTS. Falls back to truncation on error."""
        try:
            return await self._summarize(text)
        except Exception as exc:
            log.warning("Digest summarization failed, truncating: %s", exc)
            return self._truncate(text)

    async def _summarize(self, text: str) -> str:
        """LLM-powered summarization. Override or mock in tests."""
        # Default implementation: simple extractive summary
        # In production, this would call Claude Haiku for summarization
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) <= 3:
            return text
        # Keep first and last sentences + any with drama keywords
        kept = [sentences[0]]
        for s in sentences[1:-1]:
            words = set(_WORD_RE.findall(s.lower()))
            if words & _DRAMA_KEYWORDS:
                kept.append(s)
        kept.append(sentences[-1])
        return ". ".join(kept) + "."

    def _truncate(self, text: str, max_chars: int = 400) -> str:
        """Fallback truncation at sentence boundary."""
        if len(text) <= max_chars:
            return text
        # Find last sentence boundary before max_chars
        truncated = text[:max_chars]
        for delim in [".", "!", "?"]:
            idx = truncated.rfind(delim)
            if idx > 0:
                return truncated[: idx + 1]
        return truncated + "..."
