"""Loads recipes.yaml and validates against known camera presets / layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sidequest_daemon.media.recipes import CameraPreset, Recipe

_ALLOWED_CASCADE_LAYERS = {"GENRE", "WORLD", "CULTURE"}
_DYNAMIC_CAMERA_MARKER = "{camera}"


class RecipeLoader:
    def __init__(self, recipes: dict[str, Recipe]) -> None:
        self.recipes = recipes

    @classmethod
    def from_file(cls, path: Path) -> "RecipeLoader":
        return cls.from_dict(yaml.safe_load(path.read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipeLoader":
        recipes: dict[str, Recipe] = {}
        known_cameras = {p.value for p in CameraPreset}
        for name, raw in data.items():
            recipe = Recipe.model_validate(raw)

            if recipe.direction_camera != _DYNAMIC_CAMERA_MARKER:
                if recipe.direction_camera not in known_cameras:
                    raise ValueError(
                        f"recipe {name!r}: unknown camera preset "
                        f"{recipe.direction_camera!r}",
                    )

            unknown_layers = set(recipe.art_sensibility) - _ALLOWED_CASCADE_LAYERS
            if unknown_layers:
                raise ValueError(
                    f"recipe {name!r}: unknown cascade layer(s) "
                    f"{sorted(unknown_layers)}",
                )

            recipes[name] = recipe
        return cls(recipes)

    def get(self, kind: str) -> Recipe:
        return self.recipes[kind]
