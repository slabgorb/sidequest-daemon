"""Tests for Z-Image tier configuration table."""

from sidequest_daemon.media.zimage_config import (
    ZIMAGE_MODEL_VARIANT,
    ZIMAGE_QUANTIZE,
    ZIMAGE_SUPPORTED_TIERS,
    ZIMAGE_TIER_CONFIGS,
    ZImageTierConfig,
)
from sidequest_daemon.renderer.models import RenderTier


def test_every_render_tier_has_a_config():
    """Every value in RenderTier must have a ZImageTierConfig entry."""
    for tier in RenderTier:
        assert tier in ZIMAGE_TIER_CONFIGS, f"Missing config for {tier!r}"


def test_supported_tiers_matches_config_keys():
    assert ZIMAGE_SUPPORTED_TIERS == frozenset(ZIMAGE_TIER_CONFIGS)


def test_tier_config_shape():
    for tier, cfg in ZIMAGE_TIER_CONFIGS.items():
        assert isinstance(cfg, ZImageTierConfig)
        assert cfg.steps > 0
        assert cfg.guidance >= 0.0
        assert cfg.width > 0 and cfg.height > 0


def test_turbo_migration_constants_locked():
    """Lock-in: Z-Image Turbo migration (2026-04-26) values, guards rollback."""
    assert ZIMAGE_MODEL_VARIANT == "z-image-turbo"
    assert ZIMAGE_QUANTIZE == 8


def test_every_tier_uses_8_step_turbo_preset():
    """Every tier must run 8 steps + guidance=0.0 (Turbo is distilled)."""
    for tier, cfg in ZIMAGE_TIER_CONFIGS.items():
        assert cfg.steps == 8, f"{tier!r} must use 8 steps for Turbo"
        assert cfg.guidance == 0.0, f"{tier!r} must disable guidance for Turbo"


def test_worker_tier_configs_match_module_table():
    """The worker's TIER_CONFIGS dict must mirror ZIMAGE_TIER_CONFIGS.

    The worker subprocess can't import sidequest_daemon, so it duplicates
    the table — this wiring test ensures the two stay in sync.
    """
    from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker

    for tier, cfg in ZIMAGE_TIER_CONFIGS.items():
        worker_cfg = ZImageMLXWorker.TIER_CONFIGS[tier.value]
        assert worker_cfg["steps"] == cfg.steps, f"{tier!r} steps drift"
        assert worker_cfg["guidance"] == cfg.guidance, f"{tier!r} guidance drift"
        assert worker_cfg["w"] == cfg.width, f"{tier!r} width drift"
        assert worker_cfg["h"] == cfg.height, f"{tier!r} height drift"


def test_tier_config_is_frozen():
    import dataclasses
    assert dataclasses.is_dataclass(ZImageTierConfig)
    sample = next(iter(ZIMAGE_TIER_CONFIGS.values()))
    try:
        sample.steps = 999  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("ZImageTierConfig should be frozen")
