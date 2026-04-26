"""Z-Image tier configuration — maps RenderTier to generation parameters.

Importable by the main package for tests and wiring. The worker subprocess
duplicates this table (it cannot import from sidequest_daemon).

Migrated to Z-Image Turbo (Tongyi-MAI/Z-Image-Turbo) 2026-04-26 per S4-PERF
investigation. Turbo is LCM-distilled — `supports_guidance=False` in mflux's
ModelConfig, so guidance is fixed at 0.0 (CFG is a no-op for the distilled
model). Step count drops from 20 → 8, yielding the targeted ~4× speedup
(~108s → ~30s) for the chargen render path.
"""

from __future__ import annotations

from dataclasses import dataclass

from sidequest_daemon.renderer.models import RenderTier

# Model identifier passed to mflux's ModelConfig.from_name.
# `z-image-turbo` is an mflux alias for `Tongyi-MAI/Z-Image-Turbo`.
ZIMAGE_MODEL_VARIANT: str = "z-image-turbo"

# Quantization level for ZImage construction. 8-bit is the README-default
# Turbo preset and brings VRAM/cache footprint down without measurable
# quality loss on Apple Silicon.
ZIMAGE_QUANTIZE: int = 8

# Turbo is distilled — CFG is disabled on the model side. Encoded here as
# 0.0 so the value is recorded in OTEL spans and tier configs without
# claiming a guidance scale that the model does not honor.
_TURBO_GUIDANCE: float = 0.0

# 8-step preset across all tiers. Turbo's quality cliff is roughly at
# steps < 6; 8 is a safe production setting per mflux's own README example
# (`--steps 9 --quantize 8`).
_TURBO_STEPS: int = 8


@dataclass(frozen=True)
class ZImageTierConfig:
    """Z-Image generation parameters for a specific render tier."""

    steps: int
    guidance: float
    width: int
    height: int


ZIMAGE_TIER_CONFIGS: dict[RenderTier, ZImageTierConfig] = {
    RenderTier.SCENE_ILLUSTRATION: ZImageTierConfig(
        steps=_TURBO_STEPS, guidance=_TURBO_GUIDANCE, width=1024, height=768,
    ),
    RenderTier.PORTRAIT: ZImageTierConfig(
        steps=_TURBO_STEPS, guidance=_TURBO_GUIDANCE, width=768, height=1024,
    ),
    RenderTier.PORTRAIT_SQUARE: ZImageTierConfig(
        steps=_TURBO_STEPS, guidance=_TURBO_GUIDANCE, width=1024, height=1024,
    ),
    RenderTier.LANDSCAPE: ZImageTierConfig(
        steps=_TURBO_STEPS, guidance=_TURBO_GUIDANCE, width=1024, height=768,
    ),
    RenderTier.TEXT_OVERLAY: ZImageTierConfig(
        steps=_TURBO_STEPS, guidance=_TURBO_GUIDANCE, width=768, height=512,
    ),
    RenderTier.CARTOGRAPHY: ZImageTierConfig(
        steps=_TURBO_STEPS, guidance=_TURBO_GUIDANCE, width=1024, height=1024,
    ),
    RenderTier.FOG_OF_WAR: ZImageTierConfig(
        steps=_TURBO_STEPS, guidance=_TURBO_GUIDANCE, width=1024, height=1024,
    ),
}

ZIMAGE_SUPPORTED_TIERS: frozenset[RenderTier] = frozenset(ZIMAGE_TIER_CONFIGS)
