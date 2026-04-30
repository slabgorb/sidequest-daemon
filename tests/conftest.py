"""Shared pytest fixtures for sidequest-daemon tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image

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


def fake_pil_image(w: int = 64, h: int = 64) -> Image.Image:
    """Return a 64x64 black PIL Image for mock model `generate_image` returns.

    Shared between worker tests that mock the mflux model — the inference
    pipeline is not under test, only the worker's glue around it.
    """
    return Image.new("RGB", (w, h), color="black")


@pytest.fixture
def otel_exporter(monkeypatch: pytest.MonkeyPatch) -> Generator[InMemorySpanExporter, None, None]:
    """In-memory OTEL span exporter shared across daemon tests.

    Resets the global TracerProvider cleanly per test using monkeypatch on
    the `_TRACER_PROVIDER_SET_ONCE` flag — without this, OpenTelemetry
    refuses to override an already-installed provider and emits a
    'Overriding of current TracerProvider is not allowed' warning. Tests
    inspect emitted spans via ``exporter.get_finished_spans()``.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        trace,
        "_TRACER_PROVIDER_SET_ONCE",
        trace._TRACER_PROVIDER_SET_ONCE.__class__(),
    )
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()
