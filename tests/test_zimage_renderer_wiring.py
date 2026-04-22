"""Wiring test: renderer_factory returns a Z-Image SubprocessRenderer.

Per CLAUDE.md: 'Every set of tests must include at least one integration test
that verifies the component is wired into the system — imported, called, and
reachable from production code paths.'
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from sidequest_daemon.media.renderer_factory import create_renderer
from sidequest_daemon.media.renderer import SubprocessRenderer


@pytest.mark.asyncio
async def test_factory_returns_zimage_subprocess_renderer(monkeypatch):
    """With no running daemon and a GPU available, factory returns a
    SubprocessRenderer named 'zimage' that launches the zimage worker."""

    from sidequest_daemon.media import renderer_factory as rf
    from sidequest_daemon.media import gpu_detect as gd
    from sidequest_daemon.media import worker as worker_mod

    # 1) Force the daemon-not-running branch
    monkeypatch.setattr(rf, "_try_daemon", AsyncMock(return_value=None))

    # 2) Pretend we have a GPU
    class FakeGPU:
        available = True
        backend = "mlx"
        device_name = "Apple Silicon (gpu)"

    monkeypatch.setattr(rf, "detect_gpu", lambda: FakeGPU())

    # 3) Stub MediaWorker.start so we don't actually spawn a subprocess
    async def fake_start(self):
        return None

    monkeypatch.setattr(worker_mod.MediaWorker, "start", fake_start)

    renderer = await create_renderer(visual_style=None)

    assert isinstance(renderer, SubprocessRenderer)
    assert renderer.name == "zimage"


@pytest.mark.asyncio
async def test_factory_subprocess_command_targets_zimage_worker(monkeypatch):
    """Stronger check: the subprocess command invokes the zimage worker module,
    not the (deleted) flux worker or any other."""
    from sidequest_daemon.media import renderer_factory as rf
    from sidequest_daemon.media import worker as worker_mod

    monkeypatch.setattr(rf, "_try_daemon", AsyncMock(return_value=None))

    class FakeGPU:
        available = True
        backend = "mlx"
        device_name = "Apple Silicon (gpu)"

    monkeypatch.setattr(rf, "detect_gpu", lambda: FakeGPU())

    captured: dict = {}
    original_init = worker_mod.MediaWorker.__init__

    def capturing_init(self, *args, **kwargs):
        captured["name"] = kwargs.get("name")
        captured["command"] = kwargs.get("command")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(worker_mod.MediaWorker, "__init__", capturing_init)

    async def fake_start(self):
        return None

    monkeypatch.setattr(worker_mod.MediaWorker, "start", fake_start)

    await create_renderer(visual_style=None)

    assert captured["name"] == "zimage"
    cmd = captured["command"]
    assert cmd is not None
    # The command should invoke the zimage worker module via python -m
    joined = " ".join(cmd)
    assert "zimage_mlx_worker" in joined
    assert "flux_mlx_worker" not in joined
