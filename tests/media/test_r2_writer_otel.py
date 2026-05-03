"""Verify r2_writer emits start/success/failure spans."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from sidequest_daemon.media import r2_writer


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    with patch.object(r2_writer, "_tracer_provider_for_tests", provider):
        yield exp


def test_success_emits_start_and_success(exporter: InMemorySpanExporter) -> None:
    fake = MagicMock()
    with patch.object(r2_writer, "_client", lambda: fake):
        r2_writer.upload_artifact(
            world_slug="w",
            session_id="s",
            kind="portraits",
            content_bytes=b"x" * 16,
            content_type="image/png",
        )
    names = [s.name for s in exporter.get_finished_spans()]
    assert "daemon.r2.upload.start" in names
    assert "daemon.r2.upload.success" in names
    assert "daemon.r2.upload.failure" not in names


def test_failure_emits_start_and_failure(exporter: InMemorySpanExporter) -> None:
    fake = MagicMock()
    fake.put_object.side_effect = RuntimeError("simulated outage")
    with patch.object(r2_writer, "_client", lambda: fake):
        with pytest.raises(RuntimeError):
            r2_writer.upload_artifact(
                world_slug="w",
                session_id="s",
                kind="portraits",
                content_bytes=b"x" * 16,
                content_type="image/png",
            )
    names = [s.name for s in exporter.get_finished_spans()]
    assert "daemon.r2.upload.start" in names
    assert "daemon.r2.upload.failure" in names
    failure = [s for s in exporter.get_finished_spans() if s.name == "daemon.r2.upload.failure"][0]
    assert failure.attributes is not None
    assert failure.attributes["upload.error_class"] == "RuntimeError"
