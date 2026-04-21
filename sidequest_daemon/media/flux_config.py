"""Flux tier configuration — maps RenderTier to generation parameters.

Importable by the main package for tests and wiring. The worker subprocess
duplicates this table (it cannot import from sidequest).
"""

from __future__ import annotations

from dataclasses import dataclass

from sidequest_daemon.renderer.models import RenderTier


@dataclass(frozen=True)
class FluxTierConfig:
    """Flux generation parameters for a specific render tier."""

    model_variant: str
    steps: int
    guidance_scale: float
    width: int
    height: int


FLUX_TIER_CONFIGS: dict[RenderTier, FluxTierConfig] = {
    RenderTier.SCENE_ILLUSTRATION: FluxTierConfig(
        model_variant="dev",
        steps=12,
        guidance_scale=3.5,
        width=1024,
        height=768,
    ),
    RenderTier.PORTRAIT: FluxTierConfig(
        model_variant="dev",
        steps=12,
        guidance_scale=3.5,
        width=768,
        height=1024,
    ),
    RenderTier.LANDSCAPE: FluxTierConfig(
        model_variant="dev",
        steps=12,
        guidance_scale=3.5,
        width=1024,
        height=768,
    ),
    RenderTier.TEXT_OVERLAY: FluxTierConfig(
        model_variant="dev",
        steps=4,
        guidance_scale=0.0,
        width=768,
        height=512,
    ),
    RenderTier.CARTOGRAPHY: FluxTierConfig(
        model_variant="dev",
        steps=20,
        guidance_scale=3.5,
        width=1024,
        height=1024,
    ),
    RenderTier.TACTICAL_SKETCH: FluxTierConfig(
        model_variant="dev",
        steps=12,
        guidance_scale=3.5,
        width=1024,
        height=1024,
    ),
}

FLUX_SUPPORTED_TIERS: frozenset[RenderTier] = frozenset(FLUX_TIER_CONFIGS)
