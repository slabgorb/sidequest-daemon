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
from sidequest_daemon.media.recipes import (
    CameraPreset,
    LOD,
    PlaceLOD,
    RenderTarget,
)

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


def test_casting_portrait_uses_solo_description(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layers = composer._resolve_casting(t)
    assert len(layers) == 1
    assert layers[0].slot == "CASTING"
    assert layers[0].source == "npc:rux"
    assert "inquisitor" in layers[0].tokens
    assert "grey wool cassock" in layers[0].tokens  # solo is richest


def test_casting_illustration_two_uses_long(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira"], action="arguing",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    layers = composer._resolve_casting(t)
    assert len(layers) == 2
    rux = next(layer for layer in layers if layer.source == "npc:rux")
    assert rux.tokens.startswith("gaunt inquisitor in grey wool")


def test_casting_poi_uses_landmark(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    layers = composer._resolve_casting(t)
    assert len(layers) == 1
    assert layers[0].source == "where:testworld/the_lookout"
    assert "watchtower" in layers[0].tokens


def test_location_portrait_empty_when_no_background(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layers = composer._resolve_location(t)
    assert layers == []


def test_location_poi_uses_environment(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="poi", world="testworld", genre="testgenre",
        place="where:testworld/the_lookout",
    )
    layers = composer._resolve_location(t)
    assert len(layers) == 1
    assert "upland" in layers[0].tokens


def test_location_illustration_specific_uses_landmark_plus_environment(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arriving",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    layers = composer._resolve_location(t)
    combined = " ".join(layer.tokens for layer in layers)
    assert "watchtower" in combined
    assert "upland" in combined


def test_location_illustration_archetypal_uses_environment_only(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="drinking",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    layers = composer._resolve_location(t)
    combined = " ".join(layer.tokens for layer in layers)
    assert "hearth" in combined


def test_portrait_direction_action_uses_default_pose(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layer = composer._resolve_direction_action(t)
    assert "neutral expression" in layer.tokens


def test_portrait_direction_action_honors_pose_override(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
        pose_override="turning sharply, accusing gesture",
    )
    layer = composer._resolve_direction_action(t)
    assert "accusing" in layer.tokens


def test_illustration_direction_action_is_inline(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="arguing at the door",
        location="where:testgenre/tavern", camera=CameraPreset.scene,
    )
    layer = composer._resolve_direction_action(t)
    assert "arguing" in layer.tokens


def test_portrait_camera_uses_recipe_default(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layer = composer._resolve_direction_camera(t)
    assert "three-quarter" in layer.tokens


def test_illustration_camera_from_render_target(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux"], action="ambush",
        location="where:testgenre/tavern", camera=CameraPreset.topdown_90,
    )
    layer = composer._resolve_direction_camera(t)
    assert "top-down" in layer.tokens


def test_cascade_genre_world_culture_portrait(composer: PromptComposer) -> None:
    t = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    layers = composer._resolve_art_sensibility(t)
    slots = [layer.slot for layer in layers]
    assert "ART_SENSIBILITY.GENRE" in slots
    assert "ART_SENSIBILITY.WORLD" in slots
    assert "ART_SENSIBILITY.CULTURE" in slots


def test_cascade_world_empty_still_emitted_as_empty(
    composer: PromptComposer,
) -> None:
    # testworld has a visual_style; WORLD layer should have tokens.
    layers = composer._resolve_art_sensibility(
        RenderTarget(
            kind="portrait", world="testworld", genre="testgenre",
            character="npc:rux",
        ),
    )
    world_layer = next(layer for layer in layers if layer.slot == "ART_SENSIBILITY.WORLD")
    assert world_layer.tokens


def test_cascade_illustration_merges_multiple_cultures(
    composer: PromptComposer,
) -> None:
    t = RenderTarget(
        kind="illustration", world="testworld", genre="testgenre",
        participants=["npc:rux", "npc:mira"], action="standing watch",
        location="where:testworld/the_lookout", camera=CameraPreset.scene,
    )
    layers = composer._resolve_art_sensibility(t)
    culture_layers = [layer for layer in layers if layer.slot == "ART_SENSIBILITY.CULTURE"]
    # Rux and Mira share `ironhand`; dedupe produces one culture layer.
    assert len(culture_layers) == 1
    assert "iron-chased" in culture_layers[0].tokens
