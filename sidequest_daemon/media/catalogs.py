"""Catalogs — Character, Place, Style. Load at startup. Fail-loud on miss."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from sidequest_daemon.media.recipes import (
    LOD,
    CatalogMissError,
    PlaceLOD,  # noqa: F401
)


class CharacterTokens(BaseModel):
    kind: Literal["npc", "pc"]
    descriptions: dict[LOD, str]
    default_pose: str
    culture: str | None
    world: str


class CharacterCatalog:
    """World-scoped — every character belongs to exactly one world."""

    def __init__(self, entries: dict[str, CharacterTokens]) -> None:
        self._entries = entries

    @classmethod
    def load(
        cls,
        genre_packs_root: Path,
        *,
        genre: str,
        world: str,
    ) -> "CharacterCatalog":
        path = (
            genre_packs_root
            / genre
            / "worlds"
            / world
            / "portrait_manifest.yaml"
        )
        data = yaml.safe_load(path.read_text())
        entries: dict[str, CharacterTokens] = {}
        for raw in data.get("characters", []):
            slug = raw["id"]
            descriptions = {
                LOD(k): v
                for k, v in raw.get("descriptions", {}).items()
            }
            if set(descriptions) != set(LOD):
                missing = set(LOD) - set(descriptions)
                raise ValueError(
                    f"character {slug!r} missing LODs: {sorted(m.value for m in missing)}",
                )
            entries[f"npc:{slug}"] = CharacterTokens(
                kind="npc",
                descriptions=descriptions,
                default_pose=raw.get("default_pose", ""),
                culture=raw.get("culture"),
                world=world,
            )
        return cls(entries)

    def get(self, ref: str) -> CharacterTokens:
        if not (ref.startswith("npc:") or ref.startswith("pc:")):
            raise ValueError(
                f"character ref {ref!r} must use scheme 'npc:' or 'pc:'",
            )
        if ref not in self._entries:
            raise CatalogMissError(source="CharacterCatalog", missing_id=ref)
        return self._entries[ref]

    def add_pc(self, pc_id: str, tokens: CharacterTokens) -> None:
        """Register a PC at runtime from the character store."""
        self._entries[f"pc:{pc_id}"] = tokens
