"""Beat filtering — decide whether a narrative beat warrants image generation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidequest_daemon.types import GameState


# Keywords that suggest a significant NPC introduction
_INTRO_PATTERNS = re.compile(
    r"\b(emerges?|appears?|introduces?|announces?|steps? (forward|out|into)|"
    r"a (\w+ )?figure|stranger|cloaked|mysterious)\b",
    re.IGNORECASE,
)

# Action-heavy beats worth rendering (combat moves, dramatic moments)
_ACTION_PATTERNS = re.compile(
    r"\b(attacks?|strikes?|dodges?|blocks?|leaps?|charges?|explodes?|"
    r"crashes?|collapses?|transforms?|erupts?|shatters?|"
    r"draws? (a |the )?(sword|weapon|blade|bow|staff)|"
    r"flames?|lightning|thunder|earthquake|tidal|"
    r"runs?|flees?|chases?|pursues?|ambush)\b",
    re.IGNORECASE,
)

# Keywords that indicate inventory/mundane/meta actions (skip these)
_SKIP_PATTERNS = re.compile(
    r"\b(inventory|add(ed|s)? .* to (your|the) inventory|picks? up|"
    r"puts? (away|down)|stores?|equips?|orders? another|"
    r"save point|current scene state|resume from|stopping point|"
    r"your character|your stats|game saved|progress saved)\b",
    re.IGNORECASE,
)

# Dialogue-heavy text: mostly quoted speech with little action
_DIALOGUE_RATIO_THRESHOLD = 0.5


def should_generate(
    text: str,
    state: GameState,
    previous_location: str | None,
) -> bool:
    """Decide whether a narrative beat should trigger image generation.

    Default-deny: only beats matching an explicit allow pattern trigger renders.
    Deny checks run first so skip patterns are never overridden by allows.

    Allow: location changes, combat, significant NPC introductions.
    Deny: empty text, inventory/meta, dialogue-heavy, everything else.
    """
    # --- Empty guard ---
    if not text or not text.strip():
        return False

    # --- Priority allows (override deny patterns) ---

    # Combat — always generate, even if text contains skip keywords
    if state.combat.in_combat:
        return True

    # Chase — always generate during active chases
    if state.chase.in_chase:
        return True

    # Location change — always generate
    if previous_location is not None and state.location != previous_location:
        return True

    # First scene (no previous location yet) — render the opening
    if previous_location is None and state.location:
        return True

    # --- Deny checks ---

    # Skip inventory management and meta text
    if _SKIP_PATTERNS.search(text):
        return False

    # Skip dialogue-heavy beats (more than half the text is quoted speech)
    quoted = sum(len(m.group()) for m in re.finditer(r'"[^"]*"', text))
    if len(text) > 0 and quoted / len(text) > _DIALOGUE_RATIO_THRESHOLD:
        return False

    # --- Secondary allows ---

    # Significant NPC introduction
    if _INTRO_PATTERNS.search(text):
        return True

    # Action-heavy beat (combat moves, dramatic moments)
    if _ACTION_PATTERNS.search(text):
        return True

    # Default-deny: no explicit allow pattern matched
    return False
