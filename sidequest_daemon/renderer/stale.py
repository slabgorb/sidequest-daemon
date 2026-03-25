"""Stale result policy — which render results can be discarded when the player moves on."""

from __future__ import annotations

from sidequest_daemon.renderer.models import RenderResult, RenderTier

# Tiers that are low-priority enough to discard if the player has already acted.
# Text overlays are ephemeral; tactical sketches are kept (combat context matters).
_DISCARDABLE_TIERS: frozenset[RenderTier] = frozenset(
    {
        RenderTier.TEXT_OVERLAY,
        RenderTier.SCENE_ILLUSTRATION,
        RenderTier.PORTRAIT,
        RenderTier.LANDSCAPE,
        RenderTier.CARTOGRAPHY,
    }
)


def is_discardable(result: RenderResult) -> bool:
    """Return True if this result can be silently dropped when stale."""
    return result.tier in _DISCARDABLE_TIERS
