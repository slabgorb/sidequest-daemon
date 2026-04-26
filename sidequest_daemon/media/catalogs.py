"""Catalogs — Character, Place, Style. Load at startup. Fail-loud on miss."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from sidequest_daemon.media.recipes import (
    LOD,
    CatalogMissError,
    PlaceLOD,
)

log = logging.getLogger(__name__)


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


class StyleCatalog:
    def __init__(
        self,
        genre_tokens: dict[str, str],
        world_tokens: dict[tuple[str, str], str],
        culture_tokens: dict[tuple[str, str, str], str],
    ) -> None:
        self._genre = genre_tokens
        self._world = world_tokens
        self._culture = culture_tokens

    @classmethod
    def load(
        cls,
        genre_packs_root: Path,
        *,
        genre: str,
        world: str,
    ) -> "StyleCatalog":
        genre_tokens: dict[str, str] = {}
        world_tokens: dict[tuple[str, str], str] = {}
        culture_tokens: dict[tuple[str, str, str], str] = {}

        # Genre style
        genre_style = genre_packs_root / genre / "visual_style.yaml"
        if genre_style.exists():
            data = yaml.safe_load(genre_style.read_text()) or {}
            suffix = data.get("positive_suffix", "")
            if not suffix:
                # CLAUDE.md "No Silent Fallbacks": a present visual_style.yaml
                # with no usable positive_suffix is almost always a schema-
                # drift bug (e.g. legacy ``style_prompt`` / ``flux_prompt_suffix``
                # field names that the daemon never reads). Log loudly so
                # the GM panel can see why a render came out styleless.
                log.warning(
                    "style_catalog.empty_positive_suffix scope=genre genre=%s "
                    "path=%s known_keys=%s",
                    genre,
                    genre_style,
                    sorted(data.keys()),
                )
            genre_tokens[genre] = suffix

        # World style
        world_style = (
            genre_packs_root / genre / "worlds" / world / "visual_style.yaml"
        )
        if world_style.exists():
            data = yaml.safe_load(world_style.read_text()) or {}
            suffix = data.get("positive_suffix", "")
            if not suffix:
                # Bug #2a (playtest 2026-04-26): grimvault's visual_style.yaml
                # used ``style_prompt`` instead of ``positive_suffix``, so
                # world-level styling was silently dropped. Surface this as
                # a loud warning so future drift is caught at daemon startup
                # rather than discovered mid-playtest.
                log.warning(
                    "style_catalog.empty_positive_suffix scope=world "
                    "genre=%s world=%s path=%s known_keys=%s",
                    genre,
                    world,
                    world_style,
                    sorted(data.keys()),
                )
            world_tokens[(genre, world)] = suffix

        # Cultures (world-scoped — per spec)
        cultures_dir = genre_packs_root / genre / "worlds" / world / "cultures"
        if cultures_dir.is_dir():
            for culture_file in cultures_dir.glob("*.yaml"):
                data = yaml.safe_load(culture_file.read_text()) or {}
                slug = culture_file.stem
                culture_tokens[(genre, world, slug)] = data.get("visual_tokens", "")

        return cls(genre_tokens, world_tokens, culture_tokens)

    def get_genre(self, genre: str) -> str:
        if genre not in self._genre:
            raise CatalogMissError(source="StyleCatalog.genre", missing_id=genre)
        return self._genre[genre]

    def get_world(self, genre: str, world: str) -> str:
        # Absent world style is a skip-layer signal, not an error.
        return self._world.get((genre, world), "")

    def get_culture(self, genre: str, world: str, culture: str) -> str:
        key = (genre, world, culture)
        if key not in self._culture:
            raise CatalogMissError(
                source="StyleCatalog.culture",
                missing_id=f"{genre}/{world}/{culture}",
            )
        return self._culture[key]
