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
    ComposedPrompt,
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
