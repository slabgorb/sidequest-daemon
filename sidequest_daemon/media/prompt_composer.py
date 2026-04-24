"""Catalog-driven prompt composer. See spec:
docs/superpowers/specs/2026-04-24-explicit-visual-recipes-design.md
"""

from __future__ import annotations

import logging

from sidequest_daemon.media.camera_specs import CameraLoader
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import (
    CameraPreset,
    ComposedPrompt,
    LayerContribution,
    LOD,
    PlaceLOD,
    RenderTarget,
)

log = logging.getLogger(__name__)

_TOKEN_LIMIT = 512
_TOKENS_PER_WORD = 1.3
_BASE_NEGATIVES = (
    "watermark, signature, text, blurry, deformed, extra limbs, "
    "photograph, photorealistic, hyperrealistic, smooth skin, CGI"
)
_HOUSE_SAFETY_CLAUSE = "solo character focus, detailed distinctive features"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * _TOKENS_PER_WORD))


class PromptComposer:
    def __init__(
        self,
        *,
        recipes: RecipeLoader,
        cameras: CameraLoader,
        characters: CharacterCatalog,
        places: PlaceCatalog,
        styles: StyleCatalog,
    ) -> None:
        self._recipes = recipes
        self._cameras = cameras
        self._characters = characters
        self._places = places
        self._styles = styles

    def compose(self, target: RenderTarget) -> ComposedPrompt:
        raise NotImplementedError  # filled in by subsequent tasks

    def _character_lod_plan(self, target: RenderTarget) -> dict[str, LOD]:
        if target.kind == "portrait":
            assert target.character is not None
            return {target.character: LOD.SOLO}
        if target.kind == "illustration":
            participants = list(target.participants)
            n = len(participants)
            if n == 1:
                return {participants[0]: LOD.SOLO}
            if n == 2:
                return {p: LOD.LONG for p in participants}
            if 3 <= n <= 4:
                return {
                    **{participants[0]: LOD.LONG},
                    **{p: LOD.SHORT for p in participants[1:]},
                }
            # n >= 5
            return {
                participants[0]: LOD.LONG,
                participants[1]: LOD.SHORT,
                participants[2]: LOD.SHORT,
                **{p: LOD.BACKGROUND for p in participants[3:]},
            }
        return {}  # POI targets have no character plan

    def _place_lod_for(self, target: RenderTarget) -> PlaceLOD:
        if target.kind == "poi":
            return PlaceLOD.SOLO
        if target.kind == "illustration":
            return PlaceLOD.BACKDROP
        if target.kind == "portrait" and target.background:
            return PlaceLOD.BACKDROP
        return PlaceLOD.SOLO  # unreachable for current targets, safe default

    def _resolve_casting(
        self, target: RenderTarget
    ) -> list[LayerContribution]:
        if target.kind in ("portrait", "illustration"):
            plan = self._character_lod_plan(target)
            layers: list[LayerContribution] = []
            for ref, lod in plan.items():
                tokens = self._characters.get(ref)
                text = tokens.descriptions[lod]
                layers.append(
                    LayerContribution(
                        slot="CASTING",
                        source=ref,
                        tokens=text,
                        estimated_tokens=_estimate_tokens(text),
                    ),
                )
            return layers
        if target.kind == "poi":
            assert target.place is not None
            place = self._places.get(target.place)
            lod = self._place_lod_for(target)
            text = place.landmark[lod]
            return [
                LayerContribution(
                    slot="CASTING",
                    source=target.place,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        return []

    def _resolve_location(
        self, target: RenderTarget
    ) -> list[LayerContribution]:
        if target.kind == "portrait":
            if not target.background:
                return []
            place = self._places.get(target.background)
            lod = PlaceLOD.BACKDROP
            text = place.environment[lod]
            return [
                LayerContribution(
                    slot="LOCATION",
                    source=target.background,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        if target.kind == "poi":
            assert target.place is not None
            place = self._places.get(target.place)
            text = place.environment[PlaceLOD.SOLO]
            return [
                LayerContribution(
                    slot="LOCATION",
                    source=target.place,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        if target.kind == "illustration":
            assert target.location is not None
            place = self._places.get(target.location)
            lod = PlaceLOD.BACKDROP
            parts: list[str] = []
            if place.landmark[lod]:
                parts.append(place.landmark[lod])
            if place.environment[lod]:
                parts.append(place.environment[lod])
            text = ", ".join(parts)
            return [
                LayerContribution(
                    slot="LOCATION",
                    source=target.location,
                    tokens=text,
                    estimated_tokens=_estimate_tokens(text),
                ),
            ]
        return []

    def _resolve_direction_action(
        self, target: RenderTarget
    ) -> LayerContribution:
        if target.kind == "portrait":
            assert target.character is not None
            if target.pose_override:
                text = target.pose_override
                source = "inline"
            else:
                text = self._characters.get(target.character).default_pose
                source = f"{target.character}.default_pose"
        elif target.kind == "poi":
            assert target.place is not None
            place = self._places.get(target.place)
            text = place.description[PlaceLOD.SOLO]
            source = target.place
        elif target.kind == "illustration":
            text = target.action
            source = "inline"
        else:
            text, source = "", "inline"
        return LayerContribution(
            slot="DIRECTION_ACTION",
            source=source,
            tokens=text,
            estimated_tokens=_estimate_tokens(text),
        )

    def _resolve_direction_camera(
        self, target: RenderTarget
    ) -> LayerContribution:
        recipe = self._recipes.get(target.kind)
        if recipe.direction_camera == "{camera}":
            assert target.camera is not None
            preset = target.camera
        else:
            preset = CameraPreset(recipe.direction_camera)
        spec = self._cameras.get(preset)
        return LayerContribution(
            slot="DIRECTION_CAMERA",
            source=preset.value,
            tokens=spec.prompt,
            estimated_tokens=_estimate_tokens(spec.prompt),
        )
