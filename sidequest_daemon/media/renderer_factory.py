"""Renderer factory — daemon first, subprocess fallback, NullRenderer last."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sidequest_daemon.media.daemon import SOCKET_PATH
from sidequest_daemon.media.gpu_detect import detect_gpu
from sidequest_daemon.media.renderer import SubprocessRenderer
from sidequest_daemon.media.flux_config import FLUX_SUPPORTED_TIERS
from sidequest_daemon.media.worker import MediaWorker
from sidequest_daemon.renderer.base import Renderer
from sidequest_daemon.renderer.null import NullRenderer

if TYPE_CHECKING:
    from sidequest_daemon.genre.models import VisualStyle

logger = logging.getLogger(__name__)


async def _try_daemon(visual_style: "VisualStyle | None") -> Renderer | None:
    """Try connecting to a running sidequest-renderer daemon."""
    if not SOCKET_PATH.exists():
        return None

    try:
        from sidequest_daemon.media.daemon_client import DaemonClient

        client = DaemonClient(SOCKET_PATH)
        await client.start()
        logger.info("Connected to renderer daemon at %s", SOCKET_PATH)
        return SubprocessRenderer(
            worker=client,
            renderer_name="renderer-daemon",
            supported_tiers=FLUX_SUPPORTED_TIERS,
            visual_style=visual_style,
        )
    except Exception as exc:
        logger.info("Daemon not available: %s — trying subprocess", exc)
        return None


async def create_renderer(
    visual_style: "VisualStyle | None" = None,
) -> Renderer:
    """Create a renderer: daemon > subprocess > NullRenderer."""
    import sys

    # 1. Try daemon
    daemon_renderer = await _try_daemon(visual_style)
    if daemon_renderer is not None:
        return daemon_renderer

    # 2. Try subprocess (requires GPU)
    gpu = detect_gpu()
    if not gpu.available:
        logger.warning(
            "No GPU detected (backend=%s) — falling back to NullRenderer", gpu.backend
        )
        return NullRenderer()

    try:
        worker = MediaWorker(
            name="flux",
            command=[sys.executable, "-m", "sidequest.media.workers.flux_worker"],
            workdir=Path.cwd(),
            default_timeout=900.0,
        )
        await worker.start()

        return SubprocessRenderer(
            worker=worker,
            renderer_name="flux",
            supported_tiers=FLUX_SUPPORTED_TIERS,
            visual_style=visual_style,
        )
    except Exception as exc:
        logger.warning(
            "Failed to create renderer worker: %s — falling back to NullRenderer", exc
        )
        return NullRenderer()
