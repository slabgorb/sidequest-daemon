"""No-op renderer for text-only mode."""

from __future__ import annotations

from pathlib import Path

from sidequest_daemon.renderer.base import Renderer
from sidequest_daemon.renderer.models import RenderResult, RenderTier, StageCue


class NullRenderer(Renderer):
    """No-op renderer for text-only mode. Always succeeds instantly."""

    async def render(self, cue: StageCue) -> RenderResult:
        return RenderResult(
            image_path=Path("/dev/null"),
            width=0,
            height=0,
            generation_time_ms=0,
            tier=cue.tier,
            cue=cue,
            worker="null",
        )

    async def warm_up(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "null"

    def supports_tier(self, tier: RenderTier) -> bool:
        return True
