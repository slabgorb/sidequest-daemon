"""Tests for Z-Image tier configuration table."""

from sidequest_daemon.media.zimage_config import (
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


def test_tier_config_is_frozen():
    import dataclasses
    assert dataclasses.is_dataclass(ZImageTierConfig)
    sample = next(iter(ZIMAGE_TIER_CONFIGS.values()))
    try:
        sample.steps = 999  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("ZImageTierConfig should be frozen")
