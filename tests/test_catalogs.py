from pathlib import Path

import pytest

from sidequest_daemon.media.catalogs import CharacterCatalog, CharacterTokens
from sidequest_daemon.media.recipes import LOD, CatalogMissError

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
