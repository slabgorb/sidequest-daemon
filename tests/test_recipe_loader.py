from pathlib import Path

import pytest

from sidequest_daemon.media.recipe_loader import RecipeLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_loads_three_recipes():
    loader = RecipeLoader.from_file(REPO_ROOT / "recipes.yaml")
    assert set(loader.recipes) == {"portrait", "poi", "illustration"}


def test_portrait_recipe_binds_to_portrait_3q():
    loader = RecipeLoader.from_file(REPO_ROOT / "recipes.yaml")
    r = loader.recipes["portrait"]
    assert r.direction_camera == "portrait_3q"
    assert r.art_sensibility == ["GENRE", "WORLD", "CULTURE"]


def test_illustration_recipe_has_dynamic_camera():
    loader = RecipeLoader.from_file(REPO_ROOT / "recipes.yaml")
    r = loader.recipes["illustration"]
    assert r.direction_camera == "{camera}"


def test_unknown_camera_preset_rejected():
    bad = {
        "portrait": {
            "kind": "portrait",
            "casting": "character",
            "location": "background",
            "direction_action": "pose",
            "direction_camera": "fabricated_shot",
            "art_sensibility": ["GENRE"],
        },
    }
    with pytest.raises(ValueError, match="camera"):
        RecipeLoader.from_dict(bad)


def test_unknown_cascade_layer_rejected():
    bad = {
        "portrait": {
            "kind": "portrait",
            "casting": "character",
            "location": "background",
            "direction_action": "pose",
            "direction_camera": "portrait_3q",
            "art_sensibility": ["GENRE", "FABRICATED"],
        },
    }
    with pytest.raises(ValueError, match="cascade"):
        RecipeLoader.from_dict(bad)
