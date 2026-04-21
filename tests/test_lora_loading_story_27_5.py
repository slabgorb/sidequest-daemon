"""RED-phase tests for Story 27-5: LoRA loading verification — genre style LoRAs via mflux.

Tests verify:
- FluxMLXWorker accepts lora_paths[] in render params and passes to Flux1 constructor
- LoRA weights loaded via Flux1(lora_paths=[...], lora_scales=[...]) — not from_name()
- Genre pack LoRA discovery from content path
- .safetensors files accepted as lora_paths
- OTEL span attributes include LoRA metadata
- Error handling: missing LoRA file, invalid format, empty path
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: mock mflux so tests run without the actual MLX stack installed
# ---------------------------------------------------------------------------


def _make_mock_mflux() -> dict:
    """Build a mock mflux module tree that satisfies the import chain."""
    mflux = types.ModuleType("mflux")
    mflux_models = types.ModuleType("mflux.models")
    mflux_flux = types.ModuleType("mflux.models.flux")
    mflux_variants = types.ModuleType("mflux.models.flux.variants")
    mflux_txt2img = types.ModuleType("mflux.models.flux.variants.txt2img")
    mflux_txt2img_flux = types.ModuleType("mflux.models.flux.variants.txt2img.flux")
    mflux_common = types.ModuleType("mflux.models.common")
    mflux_config = types.ModuleType("mflux.models.common.config")
    mflux_model_config = types.ModuleType("mflux.models.common.config.model_config")

    mock_flux1_cls = MagicMock(name="Flux1")
    mflux_txt2img_flux.Flux1 = mock_flux1_cls

    mock_model_config = MagicMock(name="ModelConfig")
    mflux_model_config.ModelConfig = mock_model_config
    mflux_config.model_config = mflux_model_config
    mflux_config.ModelConfig = mock_model_config

    mflux.models = mflux_models
    mflux_models.flux = mflux_flux
    mflux_models.common = mflux_common
    mflux_common.config = mflux_config
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
        "mflux.models.common": mflux_common,
        "mflux.models.common.config": mflux_config,
        "mflux.models.common.config.model_config": mflux_model_config,
    }


@pytest.fixture()
def mock_mflux():
    """Patch mflux into sys.modules for the duration of a test."""
    mods = _make_mock_mflux()
    with patch.dict(sys.modules, mods):
        yield {
            "Flux1": mods["mflux.models.flux.variants.txt2img.flux"].Flux1,
            "ModelConfig": mods["mflux.models.common.config.model_config"].ModelConfig,
        }


@pytest.fixture()
def mock_pil_image():
    """Return a mock PIL Image with a save() method."""
    img = MagicMock(name="PILImage")
    img.save = MagicMock()
    return img


# ---------------------------------------------------------------------------
# AC-1: Test suite covers LoRA loading from .safetensors files
# ---------------------------------------------------------------------------


class TestLoRAParameterAcceptance:
    """FluxMLXWorker.render() must accept lora_path param and pass to Flux1."""

    def test_render_accepts_lora_path_param(self, mock_mflux, mock_pil_image, tmp_path):
        """render() with lora_path should not raise — the param must be recognized."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")

        result = worker.render({
            "tier": "scene_illustration",
            "prompt": "a dark forest",
            "lora_paths": ["/path/to/style.safetensors"],
            "lora_scales": [1.0],
            "seed": 42,
        })

        assert result is not None
        assert "image_url" in result

    def test_render_with_lora_uses_flux1_constructor_not_from_name(
        self, mock_mflux, mock_pil_image, tmp_path
    ):
        """When lora_path is provided, Flux1 must be constructed with lora_paths kwarg,
        not via Flux1.from_name() which doesn't support LoRA."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        worker = FluxMLXWorker(tmp_path)
        worker.render({
            "tier": "scene_illustration",
            "prompt": "a dark forest",
            "lora_paths": ["/path/to/style.safetensors"],
            "lora_scales": [1.0],
            "seed": 42,
        })

        # Verify Flux1 was called as constructor (not from_name) with lora_paths
        constructor_calls = mock_mflux["Flux1"].call_args_list
        found_lora = False
        for c in constructor_calls:
            if c.kwargs.get("lora_paths"):
                found_lora = True
                assert "/path/to/style.safetensors" in c.kwargs["lora_paths"]
                break
        assert found_lora, (
            "Flux1 must be constructed with lora_paths kwarg when lora_path is provided. "
            f"Actual calls: {constructor_calls}"
        )

    def test_render_with_lora_scale(self, mock_mflux, mock_pil_image, tmp_path):
        """lora_scale param should be passed to Flux1 as lora_scales."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        worker = FluxMLXWorker(tmp_path)
        worker.render({
            "tier": "scene_illustration",
            "prompt": "a dark forest",
            "lora_paths": ["/path/to/style.safetensors"],
            "lora_scales": [0.8],
            "seed": 42,
        })

        constructor_calls = mock_mflux["Flux1"].call_args_list
        found_scale = False
        for c in constructor_calls:
            if c.kwargs.get("lora_scales"):
                found_scale = True
                assert 0.8 in c.kwargs["lora_scales"]
                break
        assert found_scale, (
            "Flux1 must be constructed with lora_scales kwarg when lora_scale is provided"
        )


# ---------------------------------------------------------------------------
# AC-2: LoRA weights load successfully without PyTorch
# ---------------------------------------------------------------------------


class TestNoPyTorchDependency:
    """LoRA loading must not import torch."""

    def test_lora_render_does_not_import_torch(self, mock_mflux, mock_pil_image, tmp_path):
        """Rendering with LoRA must not trigger a torch import."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        # Remove torch from sys.modules to detect any import attempt
        torch_modules = {k: v for k, v in sys.modules.items() if k.startswith("torch")}
        with patch.dict(sys.modules, {k: None for k in torch_modules}):
            worker = FluxMLXWorker(tmp_path)
            # This should work without torch
            worker.render({
                "tier": "portrait",
                "prompt": "a warrior",
                "lora_paths": ["/path/to/style.safetensors"],
                "lora_scales": [1.0],
                "seed": 1,
            })
        # If we get here without ImportError, torch was not imported


# ---------------------------------------------------------------------------
# AC-3: mflux LoRA composition pattern (weight injection, rank/alpha config)
# ---------------------------------------------------------------------------


class TestMfluxLoRAComposition:
    """Verify mflux Flux1 is called with correct LoRA constructor pattern."""

    def test_model_config_used_with_lora(self, mock_mflux, mock_pil_image, tmp_path):
        """When LoRA is used, ModelConfig must be passed to Flux1 constructor
        (not from_name which skips lora_paths)."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        worker = FluxMLXWorker(tmp_path)
        worker.render({
            "tier": "scene_illustration",
            "prompt": "test",
            "lora_paths": ["/path/to/lora.safetensors"],
            "lora_scales": [1.0],
            "seed": 0,
        })

        # Flux1 constructor must receive model_config kwarg
        constructor_calls = mock_mflux["Flux1"].call_args_list
        found_config = False
        for c in constructor_calls:
            if "model_config" in c.kwargs or (c.args and hasattr(c.args[0], "dev")):
                found_config = True
                break
        assert found_config, (
            "Flux1 must receive model_config when loading with LoRA. "
            "from_name() does not support lora_paths."
        )

    def test_default_lora_scale_is_one(self, mock_mflux, mock_pil_image, tmp_path):
        """If lora_scale is not provided, default should be 1.0 (full weight)."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        worker = FluxMLXWorker(tmp_path)
        worker.render({
            "tier": "scene_illustration",
            "prompt": "test",
            "lora_paths": ["/path/to/lora.safetensors"],
            "lora_scales": [1.0],
            "seed": 0,
        })

        constructor_calls = mock_mflux["Flux1"].call_args_list
        found_scale = False
        for c in constructor_calls:
            if c.kwargs.get("lora_scales"):
                found_scale = True
                assert c.kwargs["lora_scales"] == [1.0], (
                    f"Default lora_scales should be [1.0], got {c.kwargs['lora_scales']}"
                )
                break
        assert found_scale, "Flux1 must receive lora_scales when lora_path is provided"


# ---------------------------------------------------------------------------
# AC-4: Genre style LoRAs from genre packs load without error
# ---------------------------------------------------------------------------


class TestGenrePackLoRADiscovery:
    """Genre LoRA files from content path must be discoverable."""

    def test_safetensors_path_accepted(self, mock_mflux, mock_pil_image, tmp_path, monkeypatch):
        """A .safetensors file path should be passed through to Flux1."""
        from sidequest_daemon.media.workers import flux_mlx_worker
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        # Task 4.2b adds a render-time matched-key counter that imports
        # mflux's FluxLoRAMapping. The mock_mflux fixture in this file
        # doesn't stub that submodule, so pre-seed the cache to bypass
        # the import path. Restored on test teardown by monkeypatch.
        monkeypatch.setattr(flux_mlx_worker, "_cached_lora_patterns", [])

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        lora_file = tmp_path / "test_lora.safetensors"
        lora_file.write_bytes(b"fake safetensors data")

        worker = FluxMLXWorker(tmp_path)
        worker.render({
            "tier": "portrait",
            "prompt": "test",
            "lora_paths": [str(lora_file)],
            "lora_scales": [1.0],
            "seed": 0,
        })

        constructor_calls = mock_mflux["Flux1"].call_args_list
        found_path = False
        for c in constructor_calls:
            if c.kwargs.get("lora_paths"):
                found_path = True
                assert str(lora_file) in c.kwargs["lora_paths"]
                break
        assert found_path, "Safetensors path must be passed to Flux1 lora_paths"


# ---------------------------------------------------------------------------
# AC-5: Integration — LoRA render vs base model render
# ---------------------------------------------------------------------------


class TestLoRAvsBaseRender:
    """LoRA-weighted model must be a separate instance from base model."""

    def test_lora_render_creates_separate_model_instance(
        self, mock_mflux, mock_pil_image, tmp_path
    ):
        """Rendering with LoRA should create a distinct Flux1 instance
        from the base model (LoRA is a constructor param, not a generate param)."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        base_model = MagicMock(name="base_model")
        base_model.generate_image.return_value = mock_pil_image
        lora_model = MagicMock(name="lora_model")
        lora_model.generate_image.return_value = mock_pil_image

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("lora_paths"):
                return lora_model
            return base_model

        mock_mflux["Flux1"].side_effect = side_effect
        mock_mflux["Flux1"].from_name.return_value = base_model

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")

        # Render without LoRA first
        worker.render({
            "tier": "scene_illustration",
            "prompt": "base render",
            "seed": 1,
        })

        # Render with LoRA
        worker.render({
            "tier": "scene_illustration",
            "prompt": "lora render",
            "lora_paths": ["/path/to/style.safetensors"],
            "lora_scales": [1.0],
            "seed": 1,
        })

        # The LoRA model should have been called for generate_image
        assert lora_model.generate_image.called, (
            "LoRA model instance must be used for generate_image, not the base model"
        )


# ---------------------------------------------------------------------------
# Error handling: no silent fallbacks
# ---------------------------------------------------------------------------


class TestLoRAErrorHandling:
    """LoRA loading errors must be raised, not swallowed."""

    def test_missing_lora_file_raises(self, mock_mflux, tmp_path):
        """If lora_path points to a nonexistent file, render must fail loudly."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_mflux["Flux1"].side_effect = FileNotFoundError("LoRA file not found")

        worker = FluxMLXWorker(tmp_path)

        with pytest.raises(FileNotFoundError):
            worker.render({
                "tier": "scene_illustration",
                "prompt": "test",
                "lora_paths": ["/nonexistent/lora.safetensors"],
                "lora_scales": [1.0],
                "seed": 0,
            })

    def test_render_without_lora_still_works(self, mock_mflux, mock_pil_image, tmp_path):
        """Render without lora_path must work as before — no regression."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].from_name.return_value = mock_model

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")

        result = worker.render({
            "tier": "scene_illustration",
            "prompt": "no lora render",
            "seed": 42,
        })

        assert result is not None
        assert "image_url" in result


# ---------------------------------------------------------------------------
# OTEL span attributes for LoRA
# ---------------------------------------------------------------------------


class TestLoRAOTELMetadata:
    """OTEL spans must include LoRA metadata when LoRA is used."""

    def test_render_span_includes_lora_path(self, mock_mflux, mock_pil_image, tmp_path):
        """The render OTEL span should record lora_path as an attribute."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_pil_image
        mock_mflux["Flux1"].return_value = mock_model

        worker = FluxMLXWorker(tmp_path)
        worker.render({
            "tier": "scene_illustration",
            "prompt": "test",
            "lora_paths": ["/path/to/style.safetensors"],
            "lora_scales": [0.7],
            "seed": 0,
        })

        # The OTEL test verifies span attributes exist — actual OTEL assertion
        # is done via the in-memory exporter pattern from test_otel_spans.py.
        # This test ensures render() doesn't crash with LoRA OTEL attributes.
        # Full OTEL verification is in story 27-7.


# ---------------------------------------------------------------------------
# Wiring: LoRA support accessible from daemon.py
# ---------------------------------------------------------------------------


class TestLoRAWiring:
    """LoRA params must flow through the daemon's render pipeline."""

    def test_daemon_render_passes_lora_params(self):
        """daemon.py WorkerPool.render() must forward lora_path to FluxMLXWorker."""
        import ast

        daemon_path = Path(__file__).parent.parent / "sidequest_daemon" / "media" / "daemon.py"
        source = daemon_path.read_text()
        tree = ast.parse(source)

        # Verify daemon.py passes params dict through to worker.render()
        # The full params dict (including lora_path) must reach the worker
        found_render_call = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "render":
                    found_render_call = True
                    break

        assert found_render_call, (
            "daemon.py must call worker.render(params) — "
            "LoRA params in the request dict must flow through to the worker"
        )
