"""SubprocessRenderer — bridges MediaWorker to the Renderer ABC."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sidequest_daemon.media.prompt_composer import PromptComposer

log = logging.getLogger(__name__)
from sidequest_daemon.media.protocol import WorkerRequest, WorkerResponse
from sidequest_daemon.media.worker import MediaWorker, WorkerError
from sidequest_daemon.renderer.base import Renderer
from sidequest_daemon.renderer.models import RenderResult, RenderTier, StageCue

if TYPE_CHECKING:
    from sidequest_daemon.genre.models import VisualStyle


class SubprocessRenderer(Renderer):
    """Renderer implementation backed by a MediaWorker subprocess."""

    def __init__(
        self,
        *,
        worker: MediaWorker,
        renderer_name: str,
        supported_tiers: frozenset[RenderTier],
        visual_style: VisualStyle | None = None,
    ) -> None:
        self._worker = worker
        self._renderer_name = renderer_name
        self._supported_tiers = supported_tiers
        self._composer: PromptComposer | None = None
        self._visual_style = visual_style
        if visual_style is not None:
            self._composer = PromptComposer(
                visual_tag_overrides=visual_style.visual_tag_overrides,
            )

    @property
    def name(self) -> str:
        return self._renderer_name

    def supports_tier(self, tier: RenderTier) -> bool:
        return tier in self._supported_tiers

    async def warm_up(self) -> None:
        from sidequest_daemon.media.protocol import WorkerRequest, WorkerStatus

        if self._worker.status == WorkerStatus.READY:
            return
        await self._worker.start()
        # Send warm_up to pre-load model and compile MPS graph
        request = WorkerRequest(method="warm_up", params={})
        await self._worker.send(request)

    async def shutdown(self) -> None:
        await self._worker.stop()

    async def render(self, cue: StageCue) -> RenderResult:
        from sidequest_daemon.media.protocol import WorkerStatus

        if self._worker.status == WorkerStatus.ERROR:
            await self._worker.start()

        params = cue.model_dump()

        # Compose prompts via PromptComposer if visual style is configured
        if self._composer is not None and self._visual_style is not None:
            composed = self._composer.compose(cue, self._visual_style)
            params["positive_prompt"] = composed.positive_prompt
            params["clip_prompt"] = composed.clip_prompt
            params["negative_prompt"] = composed.negative_prompt
            params["seed"] = composed.seed
            log.warning(
                "RENDER prompt [%s]: %s",
                cue.tier.value,
                composed.positive_prompt[:200],
            )

        request = WorkerRequest(
            method="render",
            params=params,
        )
        response: WorkerResponse = await self._worker.send(request)

        if response.error is not None:
            raise WorkerError(f"{response.error.code}: {response.error.message}")

        result = response.result
        assert result is not None  # guaranteed by protocol validation

        return RenderResult(
            image_path=Path(result["image_path"]),
            width=result["width"],
            height=result["height"],
            generation_time_ms=result.get("elapsed_ms", 0),
            tier=cue.tier,
            cue=cue,
            worker=self._renderer_name,
        )
