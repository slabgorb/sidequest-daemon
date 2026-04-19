"""Tests for FluxMLXWorker — mflux-based image generation replacing FluxWorker.

Story 27-3: Verify the MLX worker implements the same interface contract as
FluxWorker, maps parameters correctly to mflux.Flux1, handles errors loudly
(no silent fallbacks), and is wired into the daemon's WorkerPool.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: mock mflux so tests run without the actual MLX stack installed
# ---------------------------------------------------------------------------

def _make_mock_mflux() -> types.ModuleType:
    """Build a mock mflux module tree that satisfies the import chain."""
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
    """Patch mflux into sys.modules for the duration of a test."""
    mods = _make_mock_mflux()
    with patch.dict(sys.modules, mods):
        yield mods["mflux.models.flux.variants.txt2img.flux"].Flux1


@pytest.fixture()
def mock_pil_image():
    """Return a mock PIL Image with a save() method."""
    img = MagicMock(name="PILImage")
    img.save = MagicMock()
    return img


# ---------------------------------------------------------------------------
# 1. Module existence — the module must be importable
# ---------------------------------------------------------------------------

class TestModuleExists:
    """FluxMLXWorker module must exist and export the class."""

    def test_module_importable(self):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        assert FluxMLXWorker is not None

    def test_class_is_a_class(self):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        assert isinstance(FluxMLXWorker, type)


# ---------------------------------------------------------------------------
# 2. Interface contract — same methods as FluxWorker
# ---------------------------------------------------------------------------

class TestInterfaceContract:
    """FluxMLXWorker must expose the same interface as FluxWorker."""

    def test_init_accepts_output_dir(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        assert worker.output_dir == tmp_path

    def test_has_load_model(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        assert callable(getattr(worker, "load_model", None))

    def test_has_warm_up(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        assert callable(getattr(worker, "warm_up", None))

    def test_has_render(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        assert callable(getattr(worker, "render", None))

    def test_has_cleanup(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        assert callable(getattr(worker, "cleanup", None))


# ---------------------------------------------------------------------------
# 3. TIER_CONFIGS — all 6 tiers present with correct structure
# ---------------------------------------------------------------------------

class TestTierConfigs:
    """TIER_CONFIGS must match the original FluxWorker tiers."""

    EXPECTED_TIERS = {
        "scene_illustration",
        "portrait",
        "landscape",
        "text_overlay",
        "cartography",
        "tactical_sketch",
    }

    def test_all_tiers_present(self, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        assert set(FluxMLXWorker.TIER_CONFIGS.keys()) == self.EXPECTED_TIERS

    @pytest.mark.parametrize("tier", [
        "scene_illustration", "portrait", "landscape",
        "text_overlay", "cartography", "tactical_sketch",
    ])
    def test_tier_has_required_keys(self, tier, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        cfg = FluxMLXWorker.TIER_CONFIGS[tier]
        assert "model" in cfg, f"tier {tier} missing 'model'"
        assert "steps" in cfg, f"tier {tier} missing 'steps'"
        assert "guidance" in cfg, f"tier {tier} missing 'guidance'"
        assert "w" in cfg, f"tier {tier} missing 'w'"
        assert "h" in cfg, f"tier {tier} missing 'h'"

    def test_text_overlay_uses_schnell(self, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        assert FluxMLXWorker.TIER_CONFIGS["text_overlay"]["model"] == "schnell"

    def test_cartography_uses_dev(self, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        assert FluxMLXWorker.TIER_CONFIGS["cartography"]["model"] == "dev"

    def test_cartography_has_more_steps(self, mock_mflux):
        """Cartography uses 20 steps for detail — more than standard 12."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        assert FluxMLXWorker.TIER_CONFIGS["cartography"]["steps"] == 20


# ---------------------------------------------------------------------------
# 4. Init — output_dir creation and state
# ---------------------------------------------------------------------------

class TestInit:
    """Constructor must create output_dir and initialize clean state."""

    def test_creates_output_dir(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        subdir = tmp_path / "renders"
        worker = FluxMLXWorker(subdir)
        assert subdir.exists()
        assert worker.output_dir == subdir

    def test_no_models_loaded_after_init(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        # No models should be loaded until explicit load_model() call
        assert worker._active_variant is None


# ---------------------------------------------------------------------------
# 5. load_model — uses mflux.Flux1, not torch/diffusers
# ---------------------------------------------------------------------------

class TestLoadModel:
    """load_model must create Flux1 instances via mflux, not torch."""

    def test_load_schnell(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        worker.load_model("schnell")
        mock_mflux.from_name.assert_called()
        # Verify "schnell" was passed as model_name
        call_kwargs = mock_mflux.from_name.call_args
        assert call_kwargs is not None
        # Accept either positional or keyword model_name="schnell"
        args, kwargs = call_kwargs
        model_name = kwargs.get("model_name") or (args[0] if args else None)
        assert model_name == "schnell"

    def test_load_dev(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        call_kwargs = mock_mflux.from_name.call_args
        args, kwargs = call_kwargs
        model_name = kwargs.get("model_name") or (args[0] if args else None)
        assert model_name == "dev"

    def test_tracks_active_variant(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        assert worker._active_variant == "dev"


# ---------------------------------------------------------------------------
# 6. render — parameter mapping and output format
# ---------------------------------------------------------------------------

class TestRender:
    """render() must map params to Flux1.generate() and return correct format."""

    def _make_loaded_worker(self, tmp_path, mock_mflux, mock_image):
        """Create a worker with a mock model loaded."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate_image.return_value = mock_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.load_model("schnell")
        return worker, mock_instance

    def test_render_returns_image_url(self, tmp_path, mock_mflux, mock_pil_image):
        worker, mock_inst = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        result = worker.render({
            "tier": "scene_illustration",
            "positive_prompt": "a dark forest",
            "seed": 42,
        })
        assert "image_url" in result
        assert Path(result["image_url"]).suffix == ".png"

    def test_render_returns_dimensions(self, tmp_path, mock_mflux, mock_pil_image):
        worker, _ = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        result = worker.render({
            "tier": "portrait",
            "positive_prompt": "a wizard",
            "seed": 1,
        })
        assert result["width"] == 768
        assert result["height"] == 1024

    def test_render_returns_elapsed_ms(self, tmp_path, mock_mflux, mock_pil_image):
        worker, _ = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        result = worker.render({
            "tier": "landscape",
            "positive_prompt": "rolling hills",
            "seed": 7,
        })
        assert "elapsed_ms" in result
        assert isinstance(result["elapsed_ms"], int)

    def test_render_passes_correct_steps(self, tmp_path, mock_mflux, mock_pil_image):
        """Render must pass tier-specific step count to mflux.generate()."""
        worker, mock_inst = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        worker.render({
            "tier": "text_overlay",  # schnell, 4 steps
            "positive_prompt": "title card",
            "seed": 0,
        })
        call_kwargs = mock_inst.generate_image.call_args[1]
        assert call_kwargs["num_inference_steps"] == 4

    def test_render_passes_correct_dimensions(self, tmp_path, mock_mflux, mock_pil_image):
        """Render must pass tier-specific width/height to mflux.generate()."""
        worker, mock_inst = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        worker.render({
            "tier": "cartography",  # 1024x1024
            "positive_prompt": "world map",
            "seed": 0,
        })
        call_kwargs = mock_inst.generate_image.call_args[1]
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 1024

    def test_render_passes_seed(self, tmp_path, mock_mflux, mock_pil_image):
        worker, mock_inst = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        worker.render({
            "tier": "scene_illustration",
            "positive_prompt": "a castle",
            "seed": 12345,
        })
        call_kwargs = mock_inst.generate_image.call_args[1]
        assert call_kwargs["seed"] == 12345

    def test_render_passes_guidance(self, tmp_path, mock_mflux, mock_pil_image):
        worker, mock_inst = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        worker.render({
            "tier": "scene_illustration",  # guidance 3.5
            "positive_prompt": "a tower",
            "seed": 0,
        })
        call_kwargs = mock_inst.generate_image.call_args[1]
        assert call_kwargs["guidance"] == 3.5

    def test_render_saves_image_to_output_dir(self, tmp_path, mock_mflux, mock_pil_image):
        worker, _ = self._make_loaded_worker(tmp_path, mock_mflux, mock_pil_image)
        result = worker.render({
            "tier": "landscape",
            "positive_prompt": "sunset",
            "seed": 0,
        })
        # image_url must be under output_dir
        image_path = Path(result["image_url"])
        assert str(image_path).startswith(str(tmp_path))
        # PIL save() must have been called
        mock_pil_image.save.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Error handling — no silent fallbacks (CLAUDE.md rule)
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Errors must raise, never silently degrade."""

    def test_unsupported_tier_raises_valueerror(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        with pytest.raises(ValueError, match="Unsupported tier"):
            worker.render({"tier": "music", "positive_prompt": "anything"})

    def test_empty_tier_raises_valueerror(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        with pytest.raises(ValueError, match="Unsupported tier"):
            worker.render({"tier": "", "positive_prompt": "anything"})

    def test_no_prompt_content_raises_valueerror(self, tmp_path, mock_mflux, mock_pil_image):
        """render() with no prompt fields must raise, not produce garbage."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate_image.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        with pytest.raises(ValueError, match="No prompt content"):
            worker.render({"tier": "scene_illustration"})


# ---------------------------------------------------------------------------
# 8. Prompt composition — same logic as FluxWorker._compose_prompt
# ---------------------------------------------------------------------------

class TestPromptComposition:
    """Prompt composition must match FluxWorker behavior."""

    def test_positive_prompt_used_directly(self, tmp_path, mock_mflux, mock_pil_image):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate_image.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({
            "tier": "scene_illustration",
            "positive_prompt": "a specific composed prompt",
            "seed": 0,
        })
        call_kwargs = mock_instance.generate_image.call_args[1]
        assert call_kwargs["prompt"] == "a specific composed prompt"

    def test_subject_mood_location_composed(self, tmp_path, mock_mflux, mock_pil_image):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate_image.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("dev")
        worker.render({
            "tier": "scene_illustration",
            "subject": "a dragon",
            "mood": "ominous",
            "location": "mountain peak",
            "seed": 0,
        })
        call_kwargs = mock_instance.generate_image.call_args[1]
        prompt = call_kwargs["prompt"]
        assert "a dragon" in prompt
        assert "ominous" in prompt
        assert "mountain peak" in prompt

    def test_text_overlay_adds_typography_keywords(self, tmp_path, mock_mflux, mock_pil_image):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate_image.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("schnell")
        worker.render({
            "tier": "text_overlay",
            "subject": "Chapter One",
            "seed": 0,
        })
        call_kwargs = mock_instance.generate_image.call_args[1]
        prompt = call_kwargs["prompt"]
        assert "typography" in prompt.lower() or "text reading" in prompt.lower()


# ---------------------------------------------------------------------------
# 9. Cleanup — releases resources
# ---------------------------------------------------------------------------

class TestCleanup:
    """cleanup() must release all model references."""

    def test_cleanup_clears_models(self, tmp_path, mock_mflux):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        mock_mflux.from_name.return_value = MagicMock()
        worker.load_model("schnell")
        worker.cleanup()
        assert worker._active_variant is None

    def test_cleanup_idempotent(self, tmp_path, mock_mflux):
        """Calling cleanup() twice must not raise."""
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker
        worker = FluxMLXWorker(tmp_path)
        worker.cleanup()
        worker.cleanup()  # no error


# ---------------------------------------------------------------------------
# 10. No torch dependency — import hygiene (lang-review #10)
# ---------------------------------------------------------------------------

class TestNoTorchDependency:
    """FluxMLXWorker must NOT import torch at module or runtime level."""

    def test_no_torch_in_module(self, mock_mflux):
        """The module must not import torch — entire point of the migration."""
        import importlib

        # Clear any cached import
        mod_name = "sidequest_daemon.media.workers.flux_mlx_worker"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        mod = importlib.import_module(mod_name)
        source = Path(mod.__file__).read_text()
        assert "import torch" not in source, "FluxMLXWorker must not import torch"
        assert "from torch" not in source, "FluxMLXWorker must not import from torch"
        assert "from diffusers" not in source, "FluxMLXWorker must not import diffusers"


# ---------------------------------------------------------------------------
# 11. warm_up — returns timing info
# ---------------------------------------------------------------------------

class TestWarmUp:
    """warm_up() must return a dict with warmup_ms."""

    def test_warm_up_returns_warmup_ms(self, tmp_path, mock_mflux, mock_pil_image):
        from sidequest_daemon.media.workers.flux_mlx_worker import FluxMLXWorker

        mock_instance = MagicMock()
        mock_instance.generate_image.return_value = mock_pil_image
        mock_mflux.from_name.return_value = mock_instance

        worker = FluxMLXWorker(tmp_path)
        worker.load_model("schnell")
        result = worker.warm_up()
        assert "warmup_ms" in result
        assert isinstance(result["warmup_ms"], int)


# ---------------------------------------------------------------------------
# 12. Wiring test — daemon.py must import FluxMLXWorker (CLAUDE.md rule)
# ---------------------------------------------------------------------------

class TestWiring:
    """FluxMLXWorker must be wired into daemon.py WorkerPool — not just existing."""

    def test_daemon_imports_flux_mlx_worker(self):
        """daemon.py must import FluxMLXWorker, not FluxWorker."""
        source = Path(__file__).parent.parent / "sidequest_daemon" / "media" / "daemon.py"
        content = source.read_text()
        assert "FluxMLXWorker" in content, (
            "daemon.py must import FluxMLXWorker — verify wiring, not just existence"
        )

    def test_daemon_does_not_import_old_flux_worker(self):
        """daemon.py must NOT import the old FluxWorker after migration."""
        source = Path(__file__).parent.parent / "sidequest_daemon" / "media" / "daemon.py"
        content = source.read_text()
        # Check for the old import path specifically
        assert "from sidequest_daemon.media.workers.flux_worker import FluxWorker" not in content, (
            "daemon.py still imports old FluxWorker — migration incomplete"
        )


# ---------------------------------------------------------------------------
# 13. JSON-line protocol main() — subprocess entry point
# ---------------------------------------------------------------------------

class TestMainProtocol:
    """The module must have a main() function for subprocess JSON-line protocol."""

    def test_has_main_function(self, mock_mflux):
        import sidequest_daemon.media.workers.flux_mlx_worker as mod
        assert callable(getattr(mod, "main", None)), (
            "flux_mlx_worker must have a main() entry point for subprocess protocol"
        )
