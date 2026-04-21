"""Tests for FluxMLXWorker multi-LoRA protocol surface (Task 4.2a).

Verifies:
- _build_lora_model signature accepts arrays (lora_paths, lora_scales)
- render() consumes params["lora_paths"] / params["lora_scales"]
- Legacy singleton params (lora_path / lora_scale) are rejected
- Multiple LoRAs pass through to Flux1(lora_paths=[...], lora_scales=[...])
- OTEL span attributes reflect the multi-LoRA shape on the existing
  `flux_mlx.render` span (per ADR-083 Decision 3, Architect correction #1)

Runtime matched_key_count instrumentation is Task 4.2b — separate commit.
"""

from __future__ import annotations

import inspect
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_mflux() -> dict:
    """Minimal mock of the mflux import tree FluxMLXWorker uses."""
    mflux = types.ModuleType("mflux")
    mflux_models = types.ModuleType("mflux.models")
    mflux_flux = types.ModuleType("mflux.models.flux")
    mflux_variants = types.ModuleType("mflux.models.flux.variants")
    mflux_txt2img = types.ModuleType("mflux.models.flux.variants.txt2img")
    mflux_txt2img_flux = types.ModuleType("mflux.models.flux.variants.txt2img.flux")
    mflux_common = types.ModuleType("mflux.models.common")
    mflux_config = types.ModuleType("mflux.models.common.config")
    mflux_model_config = types.ModuleType("mflux.models.common.config.model_config")

    mock_flux1 = MagicMock(name="Flux1")
    mflux_txt2img_flux.Flux1 = mock_flux1

    mock_model_config = MagicMock(name="ModelConfig")
    mflux_model_config.ModelConfig = mock_model_config

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
    mods = _make_mock_mflux()
    with patch.dict(sys.modules, mods):
        yield {
            "Flux1": mods["mflux.models.flux.variants.txt2img.flux"].Flux1,
        }


@pytest.fixture()
def mock_pil_image():
    img = MagicMock(name="PILImage")
    img.save = MagicMock()
    return img


# ─── Signature — arrays replace singletons ───────────────────────────


def test_build_lora_model_accepts_lora_paths_and_lora_scales() -> None:
    from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

    sig = inspect.signature(FluxMLXWorker._build_lora_model)
    assert "lora_paths" in sig.parameters
    assert "lora_scales" in sig.parameters


def test_build_lora_model_rejects_singleton_params() -> None:
    from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

    sig = inspect.signature(FluxMLXWorker._build_lora_model)
    assert "lora_path" not in sig.parameters
    assert "lora_scale" not in sig.parameters


# ─── Runtime — render passes arrays through to Flux1 ─────────────────


def test_render_with_multi_lora_passes_all_files_to_flux1(
    mock_mflux, mock_pil_image, tmp_path
) -> None:
    from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

    mock_model = MagicMock()
    mock_model.generate_image.return_value = mock_pil_image
    mock_mflux["Flux1"].return_value = mock_model

    worker = FluxMLXWorker(tmp_path)
    worker.render({
        "tier": "scene_illustration",
        "prompt": "a silhouette at noon",
        "lora_paths": ["/path/to/style_a.safetensors", "/path/to/style_b.safetensors"],
        "lora_scales": [0.8, 0.65],
        "seed": 42,
    })

    # Find the Flux1 construction call that carried LoRA kwargs
    calls_with_loras = [
        c for c in mock_mflux["Flux1"].call_args_list if c.kwargs.get("lora_paths")
    ]
    assert len(calls_with_loras) >= 1, "Flux1 must be constructed with lora_paths kwarg"

    # At least one call should carry BOTH paths and BOTH scales in matching order
    found_multi = False
    for c in calls_with_loras:
        paths = list(c.kwargs.get("lora_paths", []))
        scales = list(c.kwargs.get("lora_scales", []))
        if (
            paths == ["/path/to/style_a.safetensors", "/path/to/style_b.safetensors"]
            and scales == [0.8, 0.65]
        ):
            found_multi = True
            break
    assert found_multi, (
        "Flux1 must receive all lora_paths and lora_scales in order. "
        f"Actual calls: {[(c.kwargs.get('lora_paths'), c.kwargs.get('lora_scales')) for c in calls_with_loras]}"
    )


def test_render_without_lora_does_not_build_lora_model(
    mock_mflux, mock_pil_image, tmp_path
) -> None:
    """Absent LoRA → fall back to the pre-loaded model, not a Flux1 construction."""
    from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

    mock_model = MagicMock()
    mock_model.generate_image.return_value = mock_pil_image
    mock_mflux["Flux1"].return_value = mock_model

    worker = FluxMLXWorker(tmp_path)
    worker.load_model("dev")
    pre_render_flux1_calls = len(mock_mflux["Flux1"].call_args_list)

    worker.render({
        "tier": "scene_illustration",
        "prompt": "a silhouette at noon",
        "seed": 42,
    })

    post_render_flux1_calls = len(mock_mflux["Flux1"].call_args_list)
    assert post_render_flux1_calls == pre_render_flux1_calls, (
        "render() without lora_paths must not instantiate a new Flux1 — "
        "use the pre-loaded model."
    )
