"""Unit tests for ZImageMLXWorker.

The ZImage model is mocked — we test worker glue, not the inference pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker


def _fake_pil_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color="black")


@pytest.fixture
def worker(tmp_path: Path) -> ZImageMLXWorker:
    # Singleton-slot reset between tests is handled by the autouse
    # fixture in conftest.py (`_reset_zimage_singleton`); this fixture
    # only constructs the worker.
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


def test_worker_targets_z_image_turbo(worker: ZImageMLXWorker):
    """Lock-in: the worker is wired to Z-Image Turbo, not base Z-Image.

    Guards against accidental rollback of the 2026-04-26 perf migration.
    """
    assert worker.MODEL_VARIANT == "z-image-turbo"
    assert worker.QUANTIZE == 8


def test_worker_uses_8_step_turbo_preset(worker: ZImageMLXWorker):
    """Lock-in: every tier uses the Turbo 8-step preset with guidance disabled."""
    for tier_name, cfg in worker.TIER_CONFIGS.items():
        assert cfg["steps"] == 8, f"{tier_name} must use 8 steps for Turbo"
        assert cfg["guidance"] == 0.0, f"{tier_name} must disable guidance for Turbo"


def test_render_calls_model_with_guidance_none_for_turbo(worker: ZImageMLXWorker):
    """Turbo's mflux ModelConfig sets supports_guidance=False; the worker
    must pass guidance=None to generate_image so we don't accidentally drive
    a CFG path on a distilled model."""
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    worker.render({"tier": "portrait", "positive_prompt": "x", "seed": 0})

    call_kwargs = mock_model.generate_image.call_args.kwargs
    assert call_kwargs["guidance"] is None
    assert call_kwargs["num_inference_steps"] == 8


def test_compose_prompt_requires_content(worker: ZImageMLXWorker):
    mock_model = MagicMock()
    mock_model.generate_image.return_value = _fake_pil_image()
    worker.model = mock_model

    with pytest.raises(ValueError, match="No prompt content"):
        worker.render({"tier": "scene_illustration", "seed": 0})


# ── Story 43-5: per-process singleton invariant ──────────────────


class TestSingletonInvariant:
    """ZImageMLXWorker is a per-process singleton (Story 43-5).

    A second construction must raise RuntimeError to fail loudly per
    CLAUDE.md "No Silent Fallbacks" — protects against any future
    revert/regression that would silently spawn a second model on the
    same MPS device.
    """

    def test_second_construction_raises(self, tmp_path: Path) -> None:
        # `worker` fixture already constructed one (which the autouse
        # fixture would normally clean up). Build the first explicitly so
        # this test is self-contained.
        ZImageMLXWorker._instance = None
        first = ZImageMLXWorker(output_dir=tmp_path / "first")
        try:
            with pytest.raises(RuntimeError, match="singleton"):
                ZImageMLXWorker(output_dir=tmp_path / "second")
        finally:
            first.cleanup()

    def test_cleanup_releases_singleton_slot(self, tmp_path: Path) -> None:
        """cleanup() must clear the singleton handle and have __init__
        repopulate it on the next construction.
        """
        ZImageMLXWorker._instance = None
        first = ZImageMLXWorker(output_dir=tmp_path / "first")
        assert ZImageMLXWorker._instance is first
        first.cleanup()
        assert ZImageMLXWorker._instance is None, (
            "cleanup() must release the singleton slot"
        )
        # Construction repopulates the slot.
        second = ZImageMLXWorker(output_dir=tmp_path / "second")
        assert ZImageMLXWorker._instance is second, (
            "__init__ must reinstall the new instance into the singleton slot"
        )
        second.cleanup()


# ── Story 43-5: contract checks for warm_up / render before load_model ──


class TestPreLoadContract:
    """`_ensure_loaded()` was removed in 43-5. `warm_up()` and `render()`
    now raise RuntimeError (not assert — `assert` is stripped under
    Python -O) when called before `load_model()`. These tests pin that
    contract; without them the assertion-replacement-with-raise change
    would have no coverage.
    """

    def test_warm_up_without_load_model_raises(
        self, worker: ZImageMLXWorker
    ) -> None:
        # Worker fixture constructs but does NOT call load_model().
        assert worker.model is None
        with pytest.raises(RuntimeError, match="warm_up.*before load_model"):
            worker.warm_up()

    def test_render_without_load_model_raises(
        self, worker: ZImageMLXWorker
    ) -> None:
        assert worker.model is None
        with pytest.raises(RuntimeError, match="render.*before load_model"):
            worker.render(
                {"tier": "scene_illustration", "positive_prompt": "x"}
            )


# ── Story 43-5: wiring proof for load_model() callers ────────────


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "sidequest_daemon"
# Canonical (and only allowed) caller of `load_model()` on an image
# worker. `WorkerPool.warm_up_image()` lives here. Any other production
# caller would bypass the singleton/idempotency guards.
_ALLOWED_LOAD_MODEL_CALLER = _SOURCE_ROOT / "media" / "daemon.py"


def test_load_model_only_called_by_workerpool() -> None:
    """Wiring proof per daemon CLAUDE.md: `load_model()` on the image
    worker must be invoked exclusively by `WorkerPool.warm_up_image` in
    `sidequest_daemon/media/daemon.py`.

    The pattern matches ANY `.load_model(` call site (regardless of how
    the caller names the handle — `worker.load_model()`, `img.load_model()`,
    `self._image.load_model()`, etc.) and also bare `load_model()` calls
    inside the worker module itself for fully-qualified safety.
    Excludes files that *define* `load_model()` (the worker class itself,
    plus the unrelated EmbedWorker._load_model which is a different
    method on a different class). The single allowed call site is the
    canonical `_ALLOWED_LOAD_MODEL_CALLER` resolved exactly by Path
    equality — no name-based or "media/" substring shortcuts that future
    sibling files could accidentally match.
    """
    # Catch any `.load_model(` invocation (call site), not declarations.
    # Declarations look like `def load_model(self) -> ...`; the leading
    # `def ` excludes them.
    call_pattern = re.compile(r"\.load_model\(")
    def_pattern = re.compile(r"^\s*def\s+_?load_model\s*\(")
    offenders: list[str] = []
    for py_file in _SOURCE_ROOT.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        if py_file.resolve() == _ALLOWED_LOAD_MODEL_CALLER.resolve():
            continue
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if def_pattern.match(line):
                continue
            if call_pattern.search(line):
                rel = py_file.relative_to(_SOURCE_ROOT.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert offenders == [], (
        "Found non-WorkerPool callers of `.load_model()` in the daemon "
        "production tree — the per-process singleton invariant assumes "
        f"`{_ALLOWED_LOAD_MODEL_CALLER.relative_to(_SOURCE_ROOT.parent)}` "
        "is the only call site:\n" + "\n".join(offenders)
    )
