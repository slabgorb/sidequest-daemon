"""Z-Image tier configuration — maps RenderTier to generation parameters.

Importable by the main package for tests and wiring. The worker subprocess
duplicates this table (it cannot import from sidequest_daemon).
"""

from __future__ import annotations

from dataclasses import dataclass

from sidequest_daemon.renderer.models import RenderTier


@dataclass(frozen=True)
class ZImageTierConfig:
    """Z-Image generation parameters for a specific render tier."""

    steps: int
    guidance: float
    width: int
    height: int


ZIMAGE_TIER_CONFIGS: dict[RenderTier, ZImageTierConfig] = {
    RenderTier.SCENE_ILLUSTRATION: ZImageTierConfig(
        steps=20, guidance=4.0, width=1024, height=768,
    ),
    RenderTier.PORTRAIT: ZImageTierConfig(
        steps=20, guidance=4.0, width=768, height=1024,
    ),
    RenderTier.PORTRAIT_SQUARE: ZImageTierConfig(
        steps=20, guidance=4.0, width=1024, height=1024,
    ),
    RenderTier.LANDSCAPE: ZImageTierConfig(
        steps=20, guidance=4.0, width=1024, height=768,
    ),
    RenderTier.TEXT_OVERLAY: ZImageTierConfig(
        steps=20, guidance=4.0, width=768, height=512,
    ),
    RenderTier.CARTOGRAPHY: ZImageTierConfig(
        steps=20, guidance=4.0, width=1024, height=1024,
    ),
    RenderTier.FOG_OF_WAR: ZImageTierConfig(
        steps=20, guidance=4.0, width=1024, height=1024,
    ),
}

ZIMAGE_SUPPORTED_TIERS: frozenset[RenderTier] = frozenset(ZIMAGE_TIER_CONFIGS)
