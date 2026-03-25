"""Abstract base class for media rendering backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sidequest_daemon.renderer.models import RenderResult, RenderTier, StageCue


class Renderer(ABC):
    """Abstract base for media rendering backends."""

    @abstractmethod
    async def render(self, cue: StageCue) -> RenderResult:
        """Generate an image from a stage cue. May be long-running."""
        ...

    @abstractmethod
    async def warm_up(self) -> None:
        """Pre-load models, allocate GPU, etc. Called once at startup."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'flux', 'null')."""
        ...

    @abstractmethod
    def supports_tier(self, tier: RenderTier) -> bool:
        """Whether this backend can handle the given tier."""
        ...
