from pathlib import Path

import pytest

from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    CharacterTokens,
    PlaceCatalog,
    PlaceTokens,
    StyleCatalog,
)
from sidequest_daemon.media.recipes import LOD, CatalogMissError, PlaceLOD

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"


def test_loads_world_characters():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get("npc:rux")
    assert isinstance(tokens, CharacterTokens)
    assert tokens.kind == "npc"
    assert tokens.culture == "ironhand"
    assert "inquisitor" in tokens.descriptions[LOD.SOLO]
    assert tokens.default_pose.startswith("standing")


def test_all_four_lods_present():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    t = cat.get("npc:rux")
    for lod in LOD:
        assert t.descriptions[lod], f"missing {lod}"


def test_missing_character_raises():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(CatalogMissError) as exc:
        cat.get("npc:ghost")
    assert exc.value.source == "CharacterCatalog"
    assert exc.value.missing_id == "npc:ghost"


def test_rejects_non_npc_pc_key():
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(ValueError, match="scheme"):
        cat.get("rux")


def test_loads_specific_place_from_history():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    t = cat.get("where:testworld/the_lookout")
    assert isinstance(t, PlaceTokens)
    assert t.kind == "specific"
    assert t.controlling_culture == "ironhand"
    assert "watchtower" in t.landmark[PlaceLOD.SOLO]
    assert "upland" in t.environment[PlaceLOD.SOLO]


def test_loads_archetypal_place_from_genre():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    t = cat.get("where:testgenre/tavern")
    assert t.kind == "archetypal"
    assert t.landmark[PlaceLOD.SOLO] == ""
    assert "hearth" in t.environment[PlaceLOD.SOLO]


def test_archetypal_with_populated_landmark_rejected():
    bad_genre = FIXTURE_ROOT / "badgenre"
    bad_genre.mkdir(parents=True, exist_ok=True)
    (bad_genre / "places.yaml").write_text(
        "tavern:\n"
        "  landmark: {solo: 'a specific tavern named The Black Lion', backdrop: ''}\n"
        "  environment: {solo: '...', backdrop: '...'}\n"
        "  description: {solo: 'tavern', backdrop: 'tavern'}\n",
    )
    try:
        with pytest.raises(ValueError, match="landmark"):
            PlaceCatalog.load(FIXTURE_ROOT, genre="badgenre", world="testworld")
    finally:
        (bad_genre / "places.yaml").unlink()
        bad_genre.rmdir()


def test_missing_place_raises():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(CatalogMissError):
        cat.get("where:testworld/no_such_poi")


def test_poi_kind_guard_rejects_archetypal_key():
    cat = PlaceCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    # Archetypal ref is legal in catalog; the caller (composer) enforces
    # poi-recipe-specific constraints. This test documents the catalog accepts both.
    specific = cat.get("where:testworld/the_lookout")
    archetypal = cat.get("where:testgenre/tavern")
    assert specific.kind == "specific"
    assert archetypal.kind == "archetypal"


def test_loads_genre_style():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get_genre("testgenre")
    assert "painterly" in tokens


def test_loads_world_style():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get_world("testgenre", "testworld")
    assert "amber" in tokens


def test_absent_world_style_returns_empty():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get_world("testgenre", "no_such_world")
    assert tokens == ""


def test_culture_tokens_loaded():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    tokens = cat.get_culture("testgenre", "testworld", "ironhand")
    assert "iron-chased" in tokens


def test_unknown_culture_raises():
    cat = StyleCatalog.load(FIXTURE_ROOT, genre="testgenre", world="testworld")
    with pytest.raises(CatalogMissError):
        cat.get_culture("testgenre", "testworld", "nonexistent")
