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


def test_loads_production_schema_characters():
    """Production manifests use flat `name` + `appearance` (no `id`, no LOD dict)."""
    cat = CharacterCatalog.load(FIXTURE_ROOT, genre="prodgenre", world="prodworld")

    # Slug derived from name by lowercasing, collapsing whitespace to `_`,
    # and dropping punctuation. "Imperatrix Celestine VII" → "imperatrix_celestine_vii".
    tokens = cat.get("npc:imperatrix_celestine_vii")
    assert isinstance(tokens, CharacterTokens)
    assert tokens.kind == "npc"

    # All four LODs populated — production has no LOD distinctions, so the
    # appearance prose populates every level. Downstream eviction/truncation
    # handles token budgeting per slot.
    for lod in LOD:
        assert tokens.descriptions[lod], f"missing {lod}"
        assert "deep brown skin" in tokens.descriptions[lod]

    # Production schema has no `culture` field; it must resolve to None
    # rather than crashing.
    assert tokens.culture is None
    # Production schema has no `default_pose`; defaults to empty string.
    assert tokens.default_pose == ""

    # Slugifier preserves hyphens and replaces only whitespace.
    two_word = cat.get("npc:two-word_name")
    assert "multi-token names" in two_word.descriptions[LOD.SOLO]


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


def test_empty_world_positive_suffix_logs_warning(
    tmp_path, caplog
) -> None:
    """Bug #2a (playtest 2026-04-26): when a world's visual_style.yaml uses
    a legacy field name (e.g. ``style_prompt`` from grimvault) and lacks a
    ``positive_suffix``, the StyleCatalog used to silently return empty
    tokens. CLAUDE.md "No Silent Fallbacks" says: log loudly so the GM
    panel and the operator see the schema-drift bug at startup rather
    than discovering it mid-playtest as styleless renders."""
    import logging

    pack = tmp_path / "drifty"
    (pack / "worlds" / "noisy").mkdir(parents=True)
    (pack / "visual_style.yaml").write_text("positive_suffix: 'genre tokens'\n")
    # World file present but uses the legacy field name — exactly the
    # grimvault failure mode.
    (pack / "worlds" / "noisy" / "visual_style.yaml").write_text(
        "style_prompt: 'this used to be silently dropped'\n"
        "negative_prompt: 'this too'\n"
    )

    with caplog.at_level(logging.WARNING, logger="sidequest_daemon.media.catalogs"):
        cat = StyleCatalog.load(tmp_path, genre="drifty", world="noisy")
    assert cat.get_world("drifty", "noisy") == ""
    assert any(
        "style_catalog.empty_positive_suffix" in record.message
        and "scope=world" in record.message
        and "world=noisy" in record.message
        for record in caplog.records
    ), (
        f"expected loud warning for empty world positive_suffix, got: "
        f"{[r.message for r in caplog.records]}"
    )
