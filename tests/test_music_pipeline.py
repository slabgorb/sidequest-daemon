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


import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch


def _write_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "task": "text2music",
        "prompt": "x",
        "audio_duration": 60,
        "actual_seeds": [42],
    }))


def test_generate_happy_path_orchestrates_all_stages(tmp_path):
    pack_dir = tmp_path / "genre_packs/cav/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    _write_json(json_path)

    # Mock adapter — pretends to write a wav at the requested path
    def fake_run(jp, output_wav):
        output_wav.write_bytes(b"fake wav bytes")
        from sidequest_daemon.media.ace_step_adapter import InferenceResult
        return InferenceResult(wav_path=output_wav, seed=42)
    adapter = MagicMock()
    adapter.run.side_effect = fake_run

    # Mock R2 uploader — records the call, returns the key
    r2_uploader = MagicMock(return_value="genre_packs/cav/audio/music/combat.ogg")

    # Mock watcher — records emits
    watcher = MagicMock()

    render_lock = asyncio.Lock()

    pipeline = MusicPipeline(
        adapter=adapter, r2_uploader=r2_uploader,
        watcher=watcher, render_lock=render_lock,
    )

    # Patch FFmpeg subprocess to just rename the wav to ogg
    with patch("sidequest_daemon.media.music_pipeline._run_ffmpeg") as mock_ffmpeg:
        def fake_ffmpeg(wav, ogg):
            ogg.write_bytes(b"fake ogg bytes")
        mock_ffmpeg.side_effect = fake_ffmpeg

        result = asyncio.run(pipeline.generate(json_path))

    assert result.r2_key == "genre_packs/cav/audio/music/combat.ogg"
    assert result.seed == 42
    adapter.run.assert_called_once()
    mock_ffmpeg.assert_called_once()
    r2_uploader.assert_called_once()
    # Watcher emitted start + complete:
    event_types = [c.args[0] for c in watcher.call_args_list]
    assert "music.generation.start" in event_types
    assert "music.generation.complete" in event_types
