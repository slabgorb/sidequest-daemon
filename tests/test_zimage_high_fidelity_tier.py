"""Story 45-38 RED — high-fidelity tier (base Z-Image 1.0, 20 steps, CFG 4.0).

Adds a fidelity dimension orthogonal to RenderTier:

  - turbo (default, in-session live narration): "z-image-turbo", 8 steps, CFG 0.0
  - high_fidelity (genre-pack pre-gen): "z-image" (base 1.0), 20 steps, CFG 4.0

These tests pin the values for AC1 (config), AC3 (turbo unchanged), and AC5
(OTEL render.prompt_composed surfaces ``model_variant`` + ``steps``).

AC4 (visual-quality eyeball test on regenerated picker_voidborn_medic_m01.png)
is NOT covered here — verifiable only by regen + comparison to the Draw Things
reference. Flagged for verify-phase manual confirmation.
"""

from __future__ import annotations

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
from sidequest_daemon.media.recipes import RenderTarget
from sidequest_daemon.renderer.models import RenderTier

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ───── AC1: high-fidelity tier values are wired into Z-Image config ─────


def test_zimage_base_model_variant_constant_exists() -> None:
    """Story 45-38 AC1: the base (non-distilled) Z-Image alias is exposed.

    Distinct from ZIMAGE_MODEL_VARIANT (which stays "z-image-turbo"). Pre-gen
    pipelines consume ZIMAGE_BASE_MODEL_VARIANT for the higher-fidelity path.
    """
    from sidequest_daemon.media import zimage_config as zc

    assert zc.ZIMAGE_BASE_MODEL_VARIANT == "z-image"


def test_high_fidelity_tier_configs_table_exists_and_covers_every_tier() -> None:
    """Every RenderTier needs an HF entry — fail loud if one is missed.

    Mirror of test_every_render_tier_has_a_config for the high-fidelity table.
    """
    from sidequest_daemon.media.zimage_config import ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS

    assert len(ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS) > 0
    for tier in RenderTier:
        assert tier in ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS, (
            f"Missing high-fidelity config for {tier!r}"
        )


def test_high_fidelity_portrait_is_base_20step_cfg4_1024sq() -> None:
    """Story 45-38 AC1: explicit values for the portrait HF tier.

    AC text: "base Z-Image 1.0, 20 steps, CFG 4, 1024x1024". The Draw Things
    reference (~/Desktop/0_painted_sci_fi_concept_art_..._3447260204.png) was
    generated at exactly these values; locking them keeps the Coyote Star
    picker portrait regen reproducible.
    """
    from sidequest_daemon.media.zimage_config import ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS

    cfg = ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS[RenderTier.PORTRAIT]
    assert cfg.steps == 20
    assert cfg.guidance == 4.0
    assert cfg.width == 1024
    assert cfg.height == 1024


def test_high_fidelity_uses_20_step_cfg4_across_all_tiers() -> None:
    """Steps + guidance are model-driven (base 1.0 supports CFG); resolution
    is tier-driven and may differ. Lock the model-driven axes on every tier.
    """
    from sidequest_daemon.media.zimage_config import ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS

    for tier, cfg in ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS.items():
        assert cfg.steps == 20, f"{tier!r} HF must use 20 steps (base 1.0)"
        assert cfg.guidance == 4.0, f"{tier!r} HF must use CFG 4.0 (base supports guidance)"


def test_zimage_tier_config_carries_model_variant() -> None:
    """ZImageTierConfig must expose model_variant so OTEL spans can surface it.

    Without this field on the config, the composer has no way to know which
    mflux alias the tier maps to, and the OTEL ``model_variant`` attribute
    on render.prompt_composed (AC5) cannot be populated.
    """
    from sidequest_daemon.media.zimage_config import (
        ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS,
        ZIMAGE_TIER_CONFIGS,
    )

    turbo_sample = ZIMAGE_TIER_CONFIGS[RenderTier.PORTRAIT]
    assert hasattr(turbo_sample, "model_variant")
    assert turbo_sample.model_variant == "z-image-turbo"

    hf_sample = ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS[RenderTier.PORTRAIT]
    assert hf_sample.model_variant == "z-image"


def test_high_fidelity_configs_all_use_base_model_variant() -> None:
    from sidequest_daemon.media.zimage_config import ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS

    for tier, cfg in ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS.items():
        assert cfg.model_variant == "z-image", (
            f"{tier!r} HF must use base 'z-image', got {cfg.model_variant!r}"
        )


# ───── AC3: Turbo path is preserved for in-session live narration ─────


def test_turbo_table_unchanged_8_step_turbo_variant() -> None:
    """AC3: in-session render path stays Turbo for latency.

    Locks the post-2026-04-26 migration values so a future "flip globally to
    base" change cannot silently regress live-narration wall-clock.
    """
    from sidequest_daemon.media.zimage_config import (
        ZIMAGE_MODEL_VARIANT,
        ZIMAGE_TIER_CONFIGS,
    )

    assert ZIMAGE_MODEL_VARIANT == "z-image-turbo"
    for tier, cfg in ZIMAGE_TIER_CONFIGS.items():
        assert cfg.steps == 8, f"Turbo {tier!r} must stay at 8 steps (live latency)"
        assert cfg.guidance == 0.0, f"Turbo {tier!r} must keep guidance disabled"
        assert cfg.model_variant == "z-image-turbo", (
            f"Turbo {tier!r} must stay on z-image-turbo"
        )


# ───── AC1/AC3 (lookup): unified accessor picks the right variant ─────


def test_get_zimage_config_default_fidelity_is_turbo() -> None:
    """In-session callers (which omit the fidelity arg) keep Turbo behavior.

    No silent fallbacks: the default is explicit, and it is Turbo.
    """
    from sidequest_daemon.media.zimage_config import get_zimage_config

    cfg = get_zimage_config(RenderTier.PORTRAIT)
    assert cfg.model_variant == "z-image-turbo"
    assert cfg.steps == 8


def test_get_zimage_config_high_fidelity_returns_base_values() -> None:
    from sidequest_daemon.media.zimage_config import get_zimage_config

    cfg = get_zimage_config(RenderTier.PORTRAIT, fidelity="high_fidelity")
    assert cfg.model_variant == "z-image"
    assert cfg.steps == 20
    assert cfg.guidance == 4.0


def test_get_zimage_config_turbo_explicit_returns_turbo_values() -> None:
    from sidequest_daemon.media.zimage_config import get_zimage_config

    cfg = get_zimage_config(RenderTier.PORTRAIT, fidelity="turbo")
    assert cfg.model_variant == "z-image-turbo"
    assert cfg.steps == 8


def test_get_zimage_config_unknown_fidelity_raises() -> None:
    """No silent fallbacks (CLAUDE.md): an unknown fidelity must fail loud."""
    from sidequest_daemon.media.zimage_config import get_zimage_config

    with pytest.raises((ValueError, KeyError)):
        get_zimage_config(RenderTier.PORTRAIT, fidelity="bogus")


# ───── AC5: render.prompt_composed OTEL span carries model_variant + steps ─────


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


def _capture_emitted_spans(monkeypatch) -> list[dict]:
    emitted: list[dict] = []

    def fake_emit(name: str, payload: dict) -> None:
        emitted.append({"name": name, "payload": payload})

    monkeypatch.setattr(
        "sidequest_daemon.media.prompt_composer._emit_watcher_event",
        fake_emit,
    )
    return emitted


def test_render_prompt_composed_emits_model_variant_and_steps_default_turbo(
    composer: PromptComposer, monkeypatch
) -> None:
    """AC5: the OTEL span for in-session (default = turbo) carries the model
    variant and step count so the GM panel can tell tiers apart at a glance.
    """
    emitted = _capture_emitted_spans(monkeypatch)

    target = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux",
    )
    composer.compose(target)

    span = next(e for e in emitted if e["name"] == "render.prompt_composed")
    assert "model_variant" in span["payload"], (
        "AC5: render.prompt_composed must include model_variant for GM panel"
    )
    assert "steps" in span["payload"], (
        "AC5: render.prompt_composed must include steps for GM panel"
    )
    assert span["payload"]["model_variant"] == "z-image-turbo"
    assert span["payload"]["steps"] == 8


def test_render_prompt_composed_emits_high_fidelity_values_when_requested(
    composer: PromptComposer, monkeypatch
) -> None:
    """AC5 wiring: when the render target requests high_fidelity, the OTEL
    span surfaces base-1.0 values (20 steps, "z-image"). End-to-end:
    RenderTarget(fidelity=...) → composer.compose → OTEL emit.
    """
    emitted = _capture_emitted_spans(monkeypatch)

    target = RenderTarget(
        kind="portrait", world="testworld", genre="testgenre",
        character="npc:rux", fidelity="high_fidelity",
    )
    composer.compose(target)

    span = next(e for e in emitted if e["name"] == "render.prompt_composed")
    assert span["payload"]["model_variant"] == "z-image"
    assert span["payload"]["steps"] == 20


def test_render_target_default_fidelity_is_turbo() -> None:
    """RenderTarget gains a fidelity field; in-session callers can omit it."""
    target = RenderTarget(
        kind="portrait", world="w", genre="g",
        character="npc:rux",
    )
    assert target.fidelity == "turbo"


def test_render_target_rejects_unknown_fidelity() -> None:
    """No silent fallbacks (CLAUDE.md): RenderTarget must reject bogus fidelity
    at construction time, not silently coerce to a default."""
    with pytest.raises(Exception):
        RenderTarget(
            kind="portrait", world="w", genre="g",
            character="npc:rux", fidelity="bogus",
        )
