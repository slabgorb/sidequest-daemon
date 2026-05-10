"""Wiring test — proves tier=music reaches MusicPipeline from a real
socket request shape. Prevents the deferral-cascade failure mode from
recurring (feature implemented, never reached)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest


@pytest.mark.asyncio
async def test_dispatch_routes_music_tier_to_music_pipeline(tmp_path):
    pack_dir = tmp_path / "genre_packs/cav/audio/music"
    pack_dir.mkdir(parents=True)
    json_path = pack_dir / "combat_input_params.json"
    json_path.write_text(json.dumps({
        "task": "text2music", "prompt": "x", "audio_duration": 60,
        "actual_seeds": [42],
    }))

    request = {
        "id": "music-test-1",
        "method": "render",
        "params": {"tier": "music", "json_params_path": str(json_path)},
    }

    from sidequest_daemon.media.music_pipeline import MusicPipeline, MusicResult

    fake_pipeline = MagicMock(spec=MusicPipeline)
    fake_pipeline.generate = AsyncMock(return_value=MusicResult(
        r2_key="genre_packs/cav/audio/music/combat.ogg",
        duration_ms=60_000, seed=42, elapsed_ms=67_000,
    ))

    from sidequest_daemon.media.daemon import dispatch_request
    reply = await dispatch_request(request, music_pipeline=fake_pipeline)

    fake_pipeline.generate.assert_called_once_with(Path(json_path))
    assert reply["result"]["r2_key"] == "genre_packs/cav/audio/music/combat.ogg"
    assert reply["result"]["seed"] == 42


@pytest.mark.asyncio
async def test_dispatch_unknown_tier_still_raises_loudly():
    """Regression: tier=foo must still raise ValueError. No silent fallback."""
    from sidequest_daemon.media.daemon import dispatch_request
    request = {
        "id": "x",
        "method": "render",
        "params": {"tier": "foo"},
    }
    with pytest.raises(ValueError, match="Unknown tier"):
        await dispatch_request(request, music_pipeline=None)
