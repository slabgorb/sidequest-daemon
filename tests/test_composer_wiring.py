"""Wiring test (per CLAUDE.md): prove daemon config loads cleanly at boot."""

from pathlib import Path

import pytest

from sidequest_daemon.media.recipes import CameraPreset
from sidequest_daemon.media.workers import zimage_mlx_worker
from sidequest_daemon.renderer.models import RenderTier, StageCue

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"


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


def test_try_compose_returns_prompt_on_success(monkeypatch) -> None:
    """try_compose_prompt_for delegates to compose_prompt_for on the happy path."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:rux",
        characters=["npc:rux"],
        camera=CameraPreset.portrait_3q,
        metadata={"world": "testworld", "genre": "testgenre"},
    )
    composed = zimage_mlx_worker.try_compose_prompt_for(cue)
    assert composed is not None
    assert "inquisitor" in composed.positive_prompt


def test_try_compose_returns_none_on_validation_error(
    monkeypatch,
    caplog,
) -> None:
    """A LANDSCAPE cue with a prose subject (not a `where:` ref) raises a
    pydantic ValidationError inside the composer; the safe wrapper must
    catch it, emit a `compose.skipped` log line, and return None so the
    daemon can fall back to the prose-subject prompt path."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.LANDSCAPE,
        # Prose, not a `where:testworld/<slug>` ref → validator rejects.
        subject="A stone tavern interior with lamplight on oak beams",
        metadata={"world": "testworld", "genre": "testgenre"},
    )

    import logging

    with caplog.at_level(
        logging.WARNING, logger="sidequest_daemon.media.workers.zimage_mlx_worker"
    ):
        composed = zimage_mlx_worker.try_compose_prompt_for(cue)

    assert composed is None
    skipped = [r for r in caplog.records if "compose.skipped" in r.getMessage()]
    assert skipped, (
        f"expected compose.skipped log, got {[r.getMessage() for r in caplog.records]}"
    )
    msg = skipped[0].getMessage()
    assert "tier=landscape" in msg
    assert "world=testgenre/testworld" in msg


def test_try_compose_returns_none_on_catalog_miss(monkeypatch, caplog) -> None:
    """A PORTRAIT cue referencing an unknown character must be caught by the
    safe wrapper (CatalogMissError) and logged, not propagated."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="npc:no_such_character",
        characters=["npc:no_such_character"],
        camera=CameraPreset.portrait_3q,
        metadata={"world": "testworld", "genre": "testgenre"},
    )

    import logging

    with caplog.at_level(
        logging.WARNING, logger="sidequest_daemon.media.workers.zimage_mlx_worker"
    ):
        composed = zimage_mlx_worker.try_compose_prompt_for(cue)

    assert composed is None
    assert any("compose.skipped" in r.getMessage() for r in caplog.records)


def test_build_cue_from_params_forwards_pc_descriptor() -> None:
    """The daemon dispatch loop projects request params into a StageCue via
    ``build_cue_from_params``. The wiring test here pins the projection: when
    the server sends ``pc_descriptor`` alongside ``world``/``genre``, the
    descriptor must land in ``cue.metadata`` so ``compose_prompt_for`` can
    register the PC at runtime. Without this projection, slice 2 is a no-op
    and every portrait keeps falling through to the prose-subject path."""
    descriptor = {
        "id": "rux",
        "appearance": "a wiry highlander in scarred leather",
        "default_pose": "hand on hilt",
        "culture": None,
    }
    params = {
        "tier": "portrait",
        "subject": "pc:rux",
        "characters": ["pc:rux"],
        "world": "flickering_reach",
        "genre": "mutant_wasteland",
        "pc_descriptor": descriptor,
        "mood": "",
        "tags": [],
        "location": "",
    }
    cue = zimage_mlx_worker.build_cue_from_params(params)
    assert cue.tier == RenderTier.PORTRAIT
    assert cue.characters == ["pc:rux"]
    assert cue.metadata["world"] == "flickering_reach"
    assert cue.metadata["genre"] == "mutant_wasteland"
    assert cue.metadata["pc_descriptor"] == descriptor


def test_build_cue_from_params_omits_descriptor_when_absent() -> None:
    """No descriptor in params → no descriptor in metadata. Slice 1 callers
    (no portrait wiring yet) must stay on the legacy path."""
    params = {
        "tier": "scene_illustration",
        "subject": "a courtyard at dusk",
        "world": "flickering_reach",
        "genre": "mutant_wasteland",
    }
    cue = zimage_mlx_worker.build_cue_from_params(params)
    assert "pc_descriptor" not in cue.metadata


def test_compose_with_pc_descriptor_registers_runtime_pc(monkeypatch) -> None:
    """A PORTRAIT cue carrying ``cue.metadata['pc_descriptor']`` must register
    the PC into the freshly-loaded CharacterCatalog before composition. This
    is slice 2 of the catalog-injected compose wiring: the server sends
    ``pc:<slug>`` refs plus a descriptor blob, and the daemon adopts the PC
    into the per-render catalog without a disk lookup. The composed prompt
    must contain the descriptor's appearance prose, proving the runtime
    registration reached the casting layer."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="pc:gladstone",
        characters=["pc:gladstone"],
        camera=CameraPreset.portrait_3q,
        metadata={
            "world": "testworld",
            "genre": "testgenre",
            "pc_descriptor": {
                "id": "gladstone",
                "appearance": "a wiry highlander in scarred leather coat, copper torc",
                "default_pose": "hand resting on belt",
                "culture": None,
            },
        },
    )
    composed = zimage_mlx_worker.compose_prompt_for(cue)
    assert "wiry highlander" in composed.positive_prompt
    # The default_pose from the descriptor must reach DIRECTION_ACTION when
    # the PORTRAIT recipe pulls it from the character's tokens.
    assert "hand resting on belt" in composed.positive_prompt


def test_try_compose_succeeds_for_pc_ref_with_descriptor(monkeypatch) -> None:
    """try_compose_prompt_for must succeed (return non-None) when the cue
    carries a pc:<slug> ref AND a matching descriptor — the catalog miss is
    avoided by the runtime add_pc path."""
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="pc:hero",
        characters=["pc:hero"],
        camera=CameraPreset.portrait_3q,
        metadata={
            "world": "testworld",
            "genre": "testgenre",
            "pc_descriptor": {
                "id": "hero",
                "appearance": "a stoic ranger in oilcloth cloak",
                "default_pose": "",
                "culture": None,
            },
        },
    )
    composed = zimage_mlx_worker.try_compose_prompt_for(cue)
    assert composed is not None
    assert "stoic ranger" in composed.positive_prompt
