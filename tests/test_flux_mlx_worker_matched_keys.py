"""Tests for ADR-083 Decision 3 Layer B — runtime matched-key visibility.

Task 4.2b: FluxMLXWorker emits `render.lora.matched_keys` (list[int])
on the existing flux_mlx.render span so the GM panel can spot adapters
that "loaded" but contribute almost nothing to the actual render.

Three layers of coverage:
1. Pure-unit:    `_count_matched_keys_for_file` against synthetic patterns
2. Drift-guard: `_validate_and_flatten_lora_patterns` API drift detection
3. Integration: render() emits `render.lora.matched_keys` of the right shape
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from safetensors.torch import save_file


# ─── Pure unit: _count_matched_keys_for_file ────────────────────────────


def test_count_matched_keys_for_file_counts_literal_pattern_hits(tmp_path: Path) -> None:
    from sidequest_daemon.media.workers.flux_mlx_worker import _count_matched_keys_for_file

    path = tmp_path / "lit.safetensors"
    save_file(
        {
            "alpha.lora_A.weight": torch.zeros(4, 4),
            "beta.lora_B.weight": torch.zeros(4, 4),
            "junk.unrelated": torch.zeros(2),
        },
        str(path),
    )
    patterns = ["alpha.lora_A.weight", "beta.lora_B.weight", "never_seen.weight"]
    assert _count_matched_keys_for_file(str(path), patterns) == 2


def test_count_matched_keys_for_file_counts_block_pattern_hits(tmp_path: Path) -> None:
    """Patterns containing {block} must resolve against numbers in the key."""
    from sidequest_daemon.media.workers.flux_mlx_worker import _count_matched_keys_for_file

    path = tmp_path / "blocks.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 4),
            "transformer_blocks.7.attn.to_q.lora_A.weight": torch.zeros(4, 4),
            "transformer_blocks.18.attn.to_q.lora_A.weight": torch.zeros(4, 4),
        },
        str(path),
    )
    patterns = ["transformer_blocks.{block}.attn.to_q.lora_A.weight"]
    assert _count_matched_keys_for_file(str(path), patterns) == 3


def test_count_matched_keys_for_file_each_key_counted_once(tmp_path: Path) -> None:
    """A key matching multiple patterns must be counted exactly once."""
    from sidequest_daemon.media.workers.flux_mlx_worker import _count_matched_keys_for_file

    path = tmp_path / "dup.safetensors"
    save_file({"shared.lora_down.weight": torch.zeros(4, 4)}, str(path))
    patterns = ["shared.lora_down.weight", "shared.lora_down.weight"]
    assert _count_matched_keys_for_file(str(path), patterns) == 1


def test_count_matched_keys_for_file_zero_for_unrecognised_keys(tmp_path: Path) -> None:
    from sidequest_daemon.media.workers.flux_mlx_worker import _count_matched_keys_for_file

    path = tmp_path / "alien.safetensors"
    save_file({"completely.alien.key": torch.zeros(4)}, str(path))
    patterns = ["transformer_blocks.{block}.attn.to_q.lora_A.weight"]
    assert _count_matched_keys_for_file(str(path), patterns) == 0


def test_count_matched_keys_for_file_zero_for_missing_file(tmp_path: Path) -> None:
    """Mirror mflux: a missing file logs and returns 0; render fails on its own."""
    from sidequest_daemon.media.workers.flux_mlx_worker import _count_matched_keys_for_file

    assert _count_matched_keys_for_file(str(tmp_path / "absent.safetensors"), ["x"]) == 0


# ─── Drift-guard: _validate_and_flatten_lora_patterns ──────────────────


def test_validate_flattens_real_mflux_mapping_into_many_patterns() -> None:
    """Sanity: real mflux produces a substantial pattern list.

    Numbers can drift as mflux versions change; the assertion is just
    that we get something plausibly large (Flux has 19 double + 38 single
    blocks × several projections × several name variants).
    """
    from sidequest_daemon.media.workers.flux_mlx_worker import (
        _validate_and_flatten_lora_patterns,
    )

    patterns = _validate_and_flatten_lora_patterns()
    assert isinstance(patterns, list)
    assert all(isinstance(p, str) for p in patterns)
    assert len(patterns) >= 100, f"only got {len(patterns)} patterns — mflux drift?"


def test_validate_raises_on_non_list_drift(monkeypatch) -> None:
    """If FluxLoRAMapping.get_mapping() ever returns a non-list, fail loud."""
    from sidequest_daemon.media.workers import flux_mlx_worker
    from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping

    monkeypatch.setattr(FluxLoRAMapping, "get_mapping", staticmethod(lambda: {"oops": 1}))
    with pytest.raises(RuntimeError, match="returned dict"):
        flux_mlx_worker._validate_and_flatten_lora_patterns()


def test_validate_raises_on_empty_mapping_drift(monkeypatch) -> None:
    from sidequest_daemon.media.workers import flux_mlx_worker
    from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping

    monkeypatch.setattr(FluxLoRAMapping, "get_mapping", staticmethod(lambda: []))
    with pytest.raises(RuntimeError, match="empty list"):
        flux_mlx_worker._validate_and_flatten_lora_patterns()


# ─── Integration: render emits the matched_keys span attribute ─────────


def _make_mock_mflux() -> dict:
    """Mock mflux import tree. Mirrors test_flux_mlx_worker_multilora.py."""
    mflux = types.ModuleType("mflux")
    mflux_models = types.ModuleType("mflux.models")
    mflux_flux = types.ModuleType("mflux.models.flux")
    mflux_variants = types.ModuleType("mflux.models.flux.variants")
    mflux_txt2img = types.ModuleType("mflux.models.flux.variants.txt2img")
    mflux_txt2img_flux = types.ModuleType("mflux.models.flux.variants.txt2img.flux")
    mflux_common = types.ModuleType("mflux.models.common")
    mflux_config = types.ModuleType("mflux.models.common.config")
    mflux_model_config = types.ModuleType("mflux.models.common.config.model_config")

    mflux_txt2img_flux.Flux1 = MagicMock(name="Flux1")
    mflux_model_config.ModelConfig = MagicMock(name="ModelConfig")

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
        yield mods["mflux.models.flux.variants.txt2img.flux"].Flux1


@pytest.fixture()
def mock_pil_image():
    img = MagicMock(name="PILImage")
    img.save = MagicMock()
    return img


@pytest.fixture()
def otel_exporter(monkeypatch):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

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


def test_render_emits_matched_keys_attribute(
    mock_mflux, mock_pil_image, otel_exporter, tmp_path, monkeypatch
) -> None:
    """When LoRA paths are passed, the flux_mlx.render span gets matched_keys."""
    from sidequest_daemon.media.workers import flux_mlx_worker
    from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

    # Inject patterns directly into the module cache — bypasses the
    # mflux validation path so this test stays decoupled from mflux's API.
    # monkeypatch restores _cached_lora_patterns on teardown.
    monkeypatch.setattr(
        flux_mlx_worker,
        "_cached_lora_patterns",
        ["transformer_blocks.{block}.attn.to_q.lora_A.weight"],
    )

    lora_a = tmp_path / "a.safetensors"
    save_file(
        {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 4)},
        str(lora_a),
    )
    lora_b = tmp_path / "b.safetensors"
    save_file({"unrecognised.key": torch.zeros(4)}, str(lora_b))

    mock_model = MagicMock()
    mock_model.generate_image.return_value = mock_pil_image
    mock_mflux.return_value = mock_model

    worker = FluxMLXWorker(tmp_path)
    worker.render({
        "tier": "scene_illustration",
        "prompt": "a town in the dust",
        "lora_paths": [str(lora_a), str(lora_b)],
        "lora_scales": [0.8, 0.5],
        "seed": 42,
    })

    spans = otel_exporter.get_finished_spans()
    render_spans = [s for s in spans if s.name == "flux_mlx.render"]
    assert render_spans, "expected a flux_mlx.render span"
    attrs = dict(render_spans[-1].attributes)

    assert "render.lora.matched_keys" in attrs, (
        f"missing attribute. Got: {sorted(attrs)}"
    )
    matched = list(attrs["render.lora.matched_keys"])
    assert matched == [1, 0], (
        f"expected [1, 0] (a matches one pattern, b matches none); got {matched}"
    )


def test_render_without_lora_omits_matched_keys_attribute(
    mock_mflux, mock_pil_image, otel_exporter, tmp_path
) -> None:
    """No LoRAs in → no matched_keys attribute (and no need to invoke counter)."""
    from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

    mock_model = MagicMock()
    mock_model.generate_image.return_value = mock_pil_image
    mock_mflux.return_value = mock_model

    worker = FluxMLXWorker(tmp_path)
    worker.load_model("dev")
    worker.render({
        "tier": "scene_illustration",
        "prompt": "naked render",
        "seed": 0,
    })

    spans = otel_exporter.get_finished_spans()
    render_spans = [s for s in spans if s.name == "flux_mlx.render"]
    attrs = dict(render_spans[-1].attributes)
    assert "render.lora.matched_keys" not in attrs


def test_render_matched_keys_length_matches_lora_paths(
    mock_mflux, mock_pil_image, otel_exporter, tmp_path, monkeypatch
) -> None:
    """One int per file, in the same order as lora_paths."""
    from sidequest_daemon.media.workers import flux_mlx_worker
    from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

    monkeypatch.setattr(flux_mlx_worker, "_cached_lora_patterns", ["only.this.weight"])

    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.safetensors"
        save_file({"only.this.weight": torch.zeros(4, 4)}, str(p))
        paths.append(str(p))

    mock_model = MagicMock()
    mock_model.generate_image.return_value = mock_pil_image
    mock_mflux.return_value = mock_model

    worker = FluxMLXWorker(tmp_path)
    worker.render({
        "tier": "landscape",
        "prompt": "a vista",
        "lora_paths": paths,
        "lora_scales": [1.0, 1.0, 1.0],
        "seed": 1,
    })

    spans = otel_exporter.get_finished_spans()
    render_spans = [s for s in spans if s.name == "flux_mlx.render"]
    attrs = dict(render_spans[-1].attributes)
    matched = list(attrs["render.lora.matched_keys"])
    assert matched == [1, 1, 1], f"expected [1, 1, 1]; got {matched}"
