"""Tests for OTEL span emission in FluxMLXWorker and gpu_detect.

Story 27-7: Verify that the MLX render pipeline emits OpenTelemetry spans
so the GM panel can observe model loads, renders, warm-ups, and GPU detection.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def otel_exporter(monkeypatch):
    """Set up an in-memory OTEL span exporter for testing.

    Uses monkeypatch to reset the global TracerProvider cleanly per test,
    avoiding the 'Overriding of current TracerProvider is not allowed' warning.
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Reset the global provider flag so set_tracer_provider works
    monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", trace._TRACER_PROVIDER_SET_ONCE.__class__())
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()


def _make_mock_mflux() -> dict:
    """Build mock mflux module tree (same as test_flux_mlx_worker.py)."""
    mflux = types.ModuleType("mflux")
    mflux_models = types.ModuleType("mflux.models")
    mflux_flux = types.ModuleType("mflux.models.flux")
    mflux_variants = types.ModuleType("mflux.models.flux.variants")
    mflux_txt2img = types.ModuleType("mflux.models.flux.variants.txt2img")
    mflux_txt2img_flux = types.ModuleType("mflux.models.flux.variants.txt2img.flux")

    mock_flux1_cls = MagicMock(name="Flux1")
    mflux_txt2img_flux.Flux1 = mock_flux1_cls
    mflux.models = mflux_models
    mflux_models.flux = mflux_flux
    mflux_flux.variants = mflux_variants
    mflux_variants.txt2img = mflux_txt2img
    mflux_txt2img.flux = mflux_txt2img_flux

    return {
        "mflux": mflux,
        "mflux.models": mflux_models,
        "mflux.models.flux": mflux_flux,
        "mflux.models.flux.variants": mflux_variants,
        "mflux.models.flux.variants.txt2img": mflux_txt2img,
        "mflux.models.flux.variants.txt2img.flux": mflux_txt2img_flux,
    }


@pytest.fixture()
def mock_mflux():
    """Patch mflux into sys.modules."""
    mods = _make_mock_mflux()
    with patch.dict(sys.modules, mods):
        yield mods["mflux.models.flux.variants.txt2img.flux"].Flux1


@pytest.fixture()
def mock_pil_image():
    """Mock PIL Image with save()."""
    img = MagicMock(name="PILImage")
    img.save = MagicMock()
    return img


# ---------------------------------------------------------------------------
# 1. render() emits a span with correct attributes
# ---------------------------------------------------------------------------

class TestRenderSpan:
    """render() must emit an OTEL span with tier, seed, dimensions, elapsed_ms."""

    def test_render_creates_span(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """render() must create a span named 'flux_mlx.render'."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({"tier": "scene_illustration", "positive_prompt": "test", "seed": 42})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        assert len(render_spans) >= 1, f"Expected 'flux_mlx.render' span, got: {[s.name for s in spans]}"

    def test_render_span_has_tier(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """render span must include tier attribute."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({"tier": "portrait", "positive_prompt": "wizard", "seed": 1})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        assert len(render_spans) >= 1
        attrs = dict(render_spans[-1].attributes)
        assert attrs.get("render.tier") == "portrait"

    def test_render_span_has_seed(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """render span must include seed attribute."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({"tier": "landscape", "positive_prompt": "hills", "seed": 999})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        assert attrs.get("render.seed") == 999

    def test_render_span_has_dimensions(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """render span must include width and height."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({"tier": "cartography", "positive_prompt": "map", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        assert attrs.get("render.width") == 1024
        assert attrs.get("render.height") == 1024

    def test_render_span_has_elapsed_ms(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """render span must include elapsed_ms attribute."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({"tier": "scene_illustration", "positive_prompt": "forest", "seed": 7})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        assert "render.elapsed_ms" in attrs
        assert isinstance(attrs["render.elapsed_ms"], int)

    def test_render_span_has_variant(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """render span must include the model variant used."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({"tier": "text_overlay", "positive_prompt": "title", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        attrs = dict(render_spans[-1].attributes)
        assert attrs.get("render.variant") == "dev"


# ---------------------------------------------------------------------------
# 2. load_model() emits a span
# ---------------------------------------------------------------------------

class TestLoadModelSpan:
    """load_model() must emit an OTEL span with variant attribute."""

    def test_load_model_creates_span(self, tmp_path, mock_mflux, otel_exporter):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")

        spans = otel_exporter.get_finished_spans()
        load_spans = [s for s in spans if s.name == "flux_mlx.load_model"]
        assert len(load_spans) >= 1, f"Expected 'flux_mlx.load_model' span, got: {[s.name for s in spans]}"

    def test_load_model_span_has_variant(self, tmp_path, mock_mflux, otel_exporter):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("schnell")

        spans = otel_exporter.get_finished_spans()
        load_spans = [s for s in spans if s.name == "flux_mlx.load_model"]
        attrs = dict(load_spans[-1].attributes)
        assert attrs.get("model.variant") == "schnell"


# ---------------------------------------------------------------------------
# 3. warm_up() emits a span
# ---------------------------------------------------------------------------

class TestWarmUpSpan:
    """warm_up() must emit an OTEL span with warmup_ms attribute."""

    def test_warm_up_creates_span(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.warm_up()

        spans = otel_exporter.get_finished_spans()
        warmup_spans = [s for s in spans if s.name == "flux_mlx.warm_up"]
        assert len(warmup_spans) >= 1, f"Expected 'flux_mlx.warm_up' span, got: {[s.name for s in spans]}"

    def test_warm_up_span_has_elapsed(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.warm_up()

        spans = otel_exporter.get_finished_spans()
        warmup_spans = [s for s in spans if s.name == "flux_mlx.warm_up"]
        attrs = dict(warmup_spans[-1].attributes)
        assert "warmup.elapsed_ms" in attrs


# ---------------------------------------------------------------------------
# 4. detect_gpu() emits a span
# ---------------------------------------------------------------------------

class TestGpuDetectSpan:
    """detect_gpu() must emit an OTEL span with backend and device info."""

    def test_detect_gpu_creates_span(self, otel_exporter):
        """detect_gpu() must create a 'gpu.detect' span."""
        from sidequest_daemon.media.gpu_detect import detect_gpu

        detect_gpu()

        spans = otel_exporter.get_finished_spans()
        gpu_spans = [s for s in spans if s.name == "gpu.detect"]
        assert len(gpu_spans) >= 1, f"Expected 'gpu.detect' span, got: {[s.name for s in spans]}"

    def test_detect_gpu_span_has_backend(self, otel_exporter):
        """gpu.detect span must include backend attribute."""
        from sidequest_daemon.media.gpu_detect import detect_gpu

        result = detect_gpu()

        spans = otel_exporter.get_finished_spans()
        gpu_spans = [s for s in spans if s.name == "gpu.detect"]
        attrs = dict(gpu_spans[-1].attributes)
        assert "gpu.backend" in attrs
        assert attrs["gpu.backend"] == result.backend

    def test_detect_gpu_span_has_available(self, otel_exporter):
        """gpu.detect span must include available attribute."""
        from sidequest_daemon.media.gpu_detect import detect_gpu

        result = detect_gpu()

        spans = otel_exporter.get_finished_spans()
        gpu_spans = [s for s in spans if s.name == "gpu.detect"]
        attrs = dict(gpu_spans[-1].attributes)
        assert "gpu.available" in attrs
        assert attrs["gpu.available"] == result.available


# ---------------------------------------------------------------------------
# 5. Error spans — render failure records exception
# ---------------------------------------------------------------------------

class TestErrorSpans:
    """Render failures must record exception on the span."""

    def test_render_error_records_exception(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """ValueError from unsupported tier should be recorded on the span."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        worker = FluxMLXWorker(tmp_path)

        with pytest.raises(ValueError):
            worker.render({"tier": "bogus", "positive_prompt": "test"})

        spans = otel_exporter.get_finished_spans()
        # Should have a render span with error status
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        assert len(render_spans) >= 1, f"Expected error span, got: {[s.name for s in spans]}"
        error_span = render_spans[-1]
        assert error_span.status.is_ok is False, "Error span should have non-OK status"


# ---------------------------------------------------------------------------
# 6. Tracer name — must use correct module name
# ---------------------------------------------------------------------------

class TestTracerName:
    """Spans must use a tracer named for the daemon media module."""

    def test_tracer_name(self, tmp_path, mock_mflux, mock_pil_image, otel_exporter):
        """Tracer instrumentation scope must identify the daemon media module."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({"tier": "scene_illustration", "positive_prompt": "test", "seed": 0})

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "flux_mlx.render"]
        assert len(render_spans) >= 1
        scope = render_spans[-1].instrumentation_scope
        assert "sidequest_daemon" in scope.name, (
            f"Tracer name should contain 'sidequest_daemon', got: {scope.name}"
        )


# ---------------------------------------------------------------------------
# 7. Wiring — OTEL dependency exists
# ---------------------------------------------------------------------------

class TestOtelWiring:
    """opentelemetry-api must be in main dependencies (not just dev)."""

    def test_otel_api_in_main_deps(self):
        """pyproject.toml must list opentelemetry-api in main dependencies."""
        source = Path(__file__).parent.parent / "pyproject.toml"
        content = source.read_text()
        # Check main dependencies section (before [project.optional-dependencies])
        main_deps_section = content.split("[project.optional-dependencies]")[0]
        assert "opentelemetry-api" in main_deps_section, (
            "opentelemetry-api must be in main dependencies, not just dev"
        )
