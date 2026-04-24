"""Catalogs — Character, Place, Style. Load at startup. Fail-loud on miss."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from sidequest_daemon.media.recipes import (
    LOD,
    CatalogMissError,
    PlaceLOD,
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
        path = genre_packs_root / genre / "worlds" / world / "portrait_manifest.yaml"
        data = yaml.safe_load(path.read_text())
        entries: dict[str, CharacterTokens] = {}
        for raw in data.get("characters", []):
            slug = raw["id"]
            descriptions = {LOD(k): v for k, v in raw.get("descriptions", {}).items()}
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


class PlaceTokens(BaseModel):
    kind: Literal["specific", "archetypal"]
    landmark: dict[PlaceLOD, str]
    environment: dict[PlaceLOD, str]
    description: dict[PlaceLOD, str]
    controlling_culture: str | None
    scope: str  # world slug (specific) or genre slug (archetypal)


class PlaceCatalog:
    def __init__(self, entries: dict[str, PlaceTokens]) -> None:
        self._entries = entries

    @classmethod
    def load(
        cls,
        genre_packs_root: Path,
        *,
        genre: str,
        world: str,
    ) -> "PlaceCatalog":
        entries: dict[str, PlaceTokens] = {}
        cls._load_specific(entries, genre_packs_root, genre, world)
        cls._load_archetypal(entries, genre_packs_root, genre)
        return cls(entries)

    @staticmethod
    def _load_specific(
        entries: dict[str, PlaceTokens],
        root: Path,
        genre: str,
        world: str,
    ) -> None:
        path = root / genre / "worlds" / world / "history.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        for chapter in data.get("chapters", []):
            for poi in chapter.get("points_of_interest", []):
                slug = poi["slug"]
                visual = poi.get("visual_prompt", {})
                env = poi.get("environment", {})
                if isinstance(visual, str) or isinstance(env, str):
                    raise ValueError(
                        f"POI {slug!r} has string visual_prompt/environment — "
                        f"migrate to {{solo: ..., backdrop: ...}} LODs "
                        f"(see scripts/migrate_poi_backdrop_lod.py)",
                    )
                landmark = {
                    PlaceLOD.SOLO: visual.get("solo", ""),
                    PlaceLOD.BACKDROP: visual.get("backdrop", ""),
                }
                environment = {
                    PlaceLOD.SOLO: env.get("solo", ""),
                    PlaceLOD.BACKDROP: env.get("backdrop", ""),
                }
                description = {
                    PlaceLOD.SOLO: poi.get("name", slug),
                    PlaceLOD.BACKDROP: poi.get("name", slug),
                }
                entries[f"where:{world}/{slug}"] = PlaceTokens(
                    kind="specific",
                    landmark=landmark,
                    environment=environment,
                    description=description,
                    controlling_culture=poi.get("controlling_culture"),
                    scope=world,
                )

    @staticmethod
    def _load_archetypal(
        entries: dict[str, PlaceTokens],
        root: Path,
        genre: str,
    ) -> None:
        path = root / genre / "places.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        for slug, raw in data.items():
            landmark = {PlaceLOD(k): v for k, v in raw["landmark"].items()}
            environment = {PlaceLOD(k): v for k, v in raw["environment"].items()}
            description = {PlaceLOD(k): v for k, v in raw["description"].items()}
            for lod, text in landmark.items():
                if text:
                    raise ValueError(
                        f"archetypal place {genre}/{slug!r} has populated "
                        f"landmark.{lod.value}; archetypes have no landmark",
                    )
            entries[f"where:{genre}/{slug}"] = PlaceTokens(
                kind="archetypal",
                landmark=landmark,
                environment=environment,
                description=description,
                controlling_culture=None,
                scope=genre,
            )

    def get(self, ref: str) -> PlaceTokens:
        if not ref.startswith("where:"):
            raise ValueError(f"place ref {ref!r} must use scheme 'where:'")
        if ref not in self._entries:
            raise CatalogMissError(source="PlaceCatalog", missing_id=ref)
        return self._entries[ref]
