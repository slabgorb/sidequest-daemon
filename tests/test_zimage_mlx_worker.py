"""Unit tests for ZImageMLXWorker.

The ZImage model is mocked — we test worker glue, not the inference pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker


def _fake_pil_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color="black")


@pytest.fixture
def worker(tmp_path: Path) -> ZImageMLXWorker:
    return ZImageMLXWorker(output_dir=tmp_path)


def test_tier_configs_match_render_tier_enum(worker: ZImageMLXWorker):
    """Worker's internal tier table must cover every tier the composer emits."""
    assert "scene_illustration" in worker.TIER_CONFIGS
    assert "portrait" in worker.TIER_CONFIGS
    assert "landscape" in worker.TIER_CONFIGS
    assert "text_overlay" in worker.TIER_CONFIGS
    assert "tactical_sketch" not in worker.TIER_CONFIGS
    assert "fog_of_war" in worker.TIER_CONFIGS
    assert "cartography" in worker.TIER_CONFIGS


def test_render_unknown_tier_raises(worker: ZImageMLXWorker):
    with pytest.raises(ValueError, match="Unsupported tier"):
        worker.render({"tier": "not_a_tier", "positive_prompt": "x"})


def test_render_rejects_lora_params(worker: ZImageMLXWorker):
    """LoRA support is removed. Passing LoRA params must fail loudly."""
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    with pytest.raises(ValueError, match="LoRA"):
        worker.render(
            {
                "tier": "scene_illustration",
                "positive_prompt": "x",
                "lora_paths": ["anything.safetensors"],
            }
        )


def test_render_returns_expected_result_shape(worker: ZImageMLXWorker):
    """Successful render returns image_url + dims + elapsed_ms."""
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    result = worker.render(
        {
            "tier": "scene_illustration",
            "positive_prompt": "a dark forest",
            "negative_prompt": "blurry",
            "seed": 42,
        }
    )

    assert "image_url" in result
    assert Path(result["image_url"]).exists()
    assert result["width"] == 1024
    assert result["height"] == 768
    assert isinstance(result["elapsed_ms"], int)


def test_render_passes_negative_prompt_to_model(worker: ZImageMLXWorker):
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    worker.render(
        {
            "tier": "portrait",
            "positive_prompt": "a face",
            "negative_prompt": "photograph, realistic",
            "seed": 1,
        }
    )

    call_kwargs = mock_model.generate_image.call_args.kwargs
    assert call_kwargs["negative_prompt"] == "photograph, realistic"
    assert call_kwargs["prompt"] == "a face"
    assert call_kwargs["seed"] == 1


def test_compose_prompt_fallback_from_raw_fields(worker: ZImageMLXWorker):
    """Batch scripts pass raw StageCue fields instead of positive_prompt."""
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    worker.render(
        {
            "tier": "portrait",
            "subject": "an old knight",
            "mood": "somber",
            "tags": ["armor", "scarred face"],
            "seed": 0,
        }
    )

    called_prompt = mock_model.generate_image.call_args.kwargs["prompt"]
    assert "an old knight" in called_prompt
    assert "somber atmosphere" in called_prompt
    assert "armor" in called_prompt


def test_compose_prompt_requires_content(worker: ZImageMLXWorker):
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    with pytest.raises(ValueError, match="No prompt content"):
        worker.render({"tier": "scene_illustration", "seed": 0})
