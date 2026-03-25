"""Stub types for cross-boundary interfaces.

These replace imports from the game engine (sidequest.game.*) that the daemon
does not ship. They provide the minimal interface the daemon code actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# DocumentEvent — replaces sidequest.game.document_event.DocumentEvent
# ---------------------------------------------------------------------------

class DocumentEvent(BaseModel):
    """A document discovered in narrative text (scroll, notice, letter, etc.)."""

    genre: str
    template_name: str
    title: str
    body_text: str
    extra_metadata: dict[str, Any] = {}
    source_agent: str | None = None


# ---------------------------------------------------------------------------
# GameState — replaces sidequest.game.state.GameState (TYPE_CHECKING only)
# ---------------------------------------------------------------------------

@dataclass
class Character:
    name: str


@dataclass
class CombatState:
    in_combat: bool = False


@dataclass
class ChaseState:
    in_chase: bool = False


@dataclass
class GameState:
    location: str = ""
    time_of_day: str = ""
    characters: list[Character] = field(default_factory=list)
    combat: CombatState = field(default_factory=CombatState)
    chase: ChaseState = field(default_factory=ChaseState)


# ---------------------------------------------------------------------------
# GenrePack protocol — replaces sidequest.genre.models.GenrePack for runtime
# ---------------------------------------------------------------------------

@runtime_checkable
class GenrePackProtocol(Protocol):
    """Minimal interface the daemon needs from a genre pack."""

    @property
    def audio(self) -> Any: ...

    @property
    def visual_style(self) -> Any: ...

    @property
    def voice_presets(self) -> Any: ...

    @property
    def required_voice_models(self) -> list[str]: ...

    @property
    def pack_dir(self) -> Any: ...


# ---------------------------------------------------------------------------
# NPC protocol — replaces sidequest.game.npc.NPC
# ---------------------------------------------------------------------------

@runtime_checkable
class HasVoice(Protocol):
    """Anything with a name and optional voice_id."""

    name: str
    voice_id: int | None
