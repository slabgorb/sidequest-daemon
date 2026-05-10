from pathlib import Path
import pytest

from sidequest_daemon.media.music_pipeline import MusicPipeline


def test_derive_r2_key_strips_input_params_suffix():
    json_path = Path("/abs/sidequest-content/genre_packs/cav/audio/music/combat_input_params.json")
    key = MusicPipeline.derive_r2_key(json_path)
    assert key == "genre_packs/cav/audio/music/combat.ogg"


def test_derive_r2_key_handles_world_subpacks():
    json_path = Path("/abs/sidequest-content/genre_packs/cav/worlds/sunden/audio/music/combat_input_params.json")
    key = MusicPipeline.derive_r2_key(json_path)
    assert key == "genre_packs/cav/worlds/sunden/audio/music/combat.ogg"


def test_derive_r2_key_rejects_path_outside_genre_packs():
    json_path = Path("/abs/elsewhere/audio/music/combat_input_params.json")
    with pytest.raises(ValueError, match="INVALID_PARAMS_LOCATION"):
        MusicPipeline.derive_r2_key(json_path)


def test_derive_r2_key_rejects_wrong_filename_suffix():
    json_path = Path("/abs/sidequest-content/genre_packs/cav/audio/music/combat.json")
    with pytest.raises(ValueError, match="INVALID_PARAMS_LOCATION"):
        MusicPipeline.derive_r2_key(json_path)
