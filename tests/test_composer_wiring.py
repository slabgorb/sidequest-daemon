"""Wiring test (per CLAUDE.md): prove daemon config loads cleanly at boot."""

from pathlib import Path

import pytest

from sidequest_daemon.media.recipes import CameraPreset
from sidequest_daemon.media.workers import zimage_mlx_worker
from sidequest_daemon.renderer.models import RenderTier, StageCue

FIXTURE_ROOT = (
    Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"
)


def test_daemon_refuses_to_start_with_invalid_recipes(tmp_path) -> None:
    from sidequest_daemon.media import daemon as daemon_module

    bad = tmp_path / "recipes.yaml"
    bad.write_text("portrait: {kind: portrait, direction_camera: fabricated_shot}")
    with pytest.raises(ValueError):
        daemon_module.validate_startup_config(
            recipes_path=bad,
            cameras_path=Path(__file__).resolve().parents[1] / "cameras.yaml",
        )


def test_daemon_accepts_valid_config() -> None:
    from sidequest_daemon.media import daemon as daemon_module

    root = Path(__file__).resolve().parents[1]
    daemon_module.validate_startup_config(
        recipes_path=root / "recipes.yaml",
        cameras_path=root / "cameras.yaml",
    )


def test_worker_imports_new_composer() -> None:
    """The worker must import PromptComposer from prompt_composer.py."""
    source = Path(zimage_mlx_worker.__file__).read_text()
    assert "from sidequest_daemon.media.prompt_composer import PromptComposer" in source
    assert "RenderTarget" in source


def test_worker_build_render_target_from_cue(monkeypatch) -> None:
    """When the worker receives a StageCue with a CameraPreset, it must
    construct a valid RenderTarget and pass it to the composer."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:rux",
        characters=["npc:rux"],
        camera=CameraPreset.portrait_3q,
        metadata={"world": "testworld", "genre": "testgenre"},
    )
    target = zimage_mlx_worker.build_render_target(cue)
    assert target.kind == "portrait"
    assert target.character == "npc:rux"
    assert target.world == "testworld"
    assert target.genre == "testgenre"


def test_wiring_end_to_end_produces_nonempty_prompt(monkeypatch) -> None:
    """The worker's compose path produces a non-empty positive prompt."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:rux",
        characters=["npc:rux"],
        camera=CameraPreset.portrait_3q,
        metadata={"world": "testworld", "genre": "testgenre"},
    )
    prompt = zimage_mlx_worker.compose_prompt_for(cue)
    assert "inquisitor" in prompt.positive_prompt
    assert prompt.seed != 0
