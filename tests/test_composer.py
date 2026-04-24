from pathlib import Path

import pytest

from sidequest_daemon.media.camera_specs import CameraLoader
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.prompt_composer import PromptComposer
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import CameraPreset, LOD, PlaceLOD, RenderTarget

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def composer() -> PromptComposer:
    return PromptComposer(
        recipes=RecipeLoader.from_file(REPO_ROOT / "recipes.yaml"),
        cameras=CameraLoader.from_file(REPO_ROOT / "cameras.yaml"),
        characters=CharacterCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
        places=PlaceCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
        styles=StyleCatalog.load(
            FIXTURE_ROOT, genre="testgenre", world="testworld"
        ),
    )


def test_composer_constructs(composer: PromptComposer) -> None:
    assert composer is not None


def test_portrait_lod_plan_is_solo(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    plan = composer._character_lod_plan(t)
    assert plan == {"npc:rux": LOD.SOLO}


def test_illustration_one_participant_is_solo(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="thinking",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    assert composer._character_lod_plan(t) == {"npc:rux": LOD.SOLO}


def test_illustration_two_participants_both_long(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira"], action="arguing",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    assert composer._character_lod_plan(t) == {
        "npc:rux": LOD.LONG,
        "npc:mira": LOD.LONG,
    }


def test_illustration_four_participants_focus_long_rest_short(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira", "npc:a", "npc:b"],
        action="around a table", location="where:testgenre/tavern",
        camera=CameraPreset.scene,
    )
    plan = composer._character_lod_plan(t)
    assert plan["npc:rux"] == LOD.LONG
    for other in ("npc:mira", "npc:a", "npc:b"):
        assert plan[other] == LOD.SHORT


def test_illustration_six_participants_tail_background(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:a", "npc:b", "npc:c", "npc:d", "npc:e"],
        action="tavern brawl", location="where:testgenre/tavern",
        camera=CameraPreset.scene,
    )
    plan = composer._character_lod_plan(t)
    assert plan["npc:rux"] == LOD.LONG
    assert plan["npc:a"] == LOD.SHORT
    assert plan["npc:b"] == LOD.SHORT
    assert plan["npc:c"] == LOD.BACKGROUND
    assert plan["npc:d"] == LOD.BACKGROUND
    assert plan["npc:e"] == LOD.BACKGROUND


def test_poi_lod_solo(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    assert composer._place_lod_for(t) == PlaceLOD.SOLO


def test_illustration_place_lod_backdrop(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arriving",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    assert composer._place_lod_for(t) == PlaceLOD.BACKDROP
