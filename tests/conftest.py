"""Shared pytest fixtures for sidequest-daemon tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker


@pytest.fixture(autouse=True)
def _reset_zimage_singleton() -> Generator[None, None, None]:
    """Story 43-5: ZImageMLXWorker is a per-process singleton.

    Reset the class-level `_instance` slot before and after every test so
    that any test which constructs a worker (directly or indirectly via
    WorkerPool) starts from a clean state. Without this, the second test
    file in any pytest run would trip the singleton guard at fixture-
    build time. (Importing the worker module does not construct an
    instance — the guard only fires on `ZImageMLXWorker(...)` calls.)
    """
    ZImageMLXWorker._instance = None
    yield
    ZImageMLXWorker._instance = None
