"""Story 45-39 RED — worker swap on SIDEQUEST_DAEMON_FIDELITY.

Story 45-38 wired ``fidelity`` from the pre-gen scripts through the daemon's
JSON-RPC params, into ``StageCue.metadata`` and ``RenderTarget.fidelity``,
and out the composer's OTEL ``render.prompt_composed`` span. But
``ZImageMLXWorker`` is a per-process singleton that loads ONE mflux model at
startup; its class-level ``MODEL_VARIANT`` and ``TIER_CONFIGS`` were
turbo-only. Result: high-fidelity render requests reached the daemon with
the right composer span but mflux still inferenced with the loaded turbo
model. AC4 of 45-38 (visibly painterly portraits) was unreachable.

This test module pins the per-init env-var contract for the worker:

  - AC1: ``SIDEQUEST_DAEMON_FIDELITY`` is read at worker construction and
    determines the loaded model + tier table.
  - AC2: ``high_fidelity`` mode pulls from ``ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS``
    (20 steps, guidance 4.0, base ``z-image`` alias).
  - AC3: A request whose ``fidelity`` does not match the daemon's loaded
    fidelity is rejected with a structured, loud error — no silent fallback
    onto the wrong model.
  - AC4: The worker's ``render.*`` and ``zimage_mlx.load_model`` OTEL spans
    surface the *loaded* variant. Composer-vs-worker divergence on the
    ``model_variant`` attribute is the diagnostic signal for misconfiguration
    (composer says HF, worker says turbo → daemon was launched with the
    wrong env var).

AC5 (manual eyeball comparison of regenerated picker portrait against the
Draw Things reference at ~/Desktop/0_painted_sci_fi_concept_art_..._3447260204.png)
is NOT covered here — verifiable only by regen + visual inspection. Flagged
for verify-phase manual confirmation by Dev/Reviewer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sidequest_daemon.media.workers.zimage_mlx_worker import ZImageMLXWorker
from tests.conftest import fake_pil_image


def _attached_mock_model(worker: ZImageMLXWorker) -> MagicMock:
    """Bypass the actual mflux load by attaching a mock model post-init.

    Tests in this module exercise worker glue (env var → tier table → mflux
    kwargs), not the inference pipeline. ``load_model()`` is the env-var
    boundary and is tested separately via OTEL spans without invoking
    mflux.
    """
    mock_model = MagicMock(name="ZImage")
    mock_model.generate_image.return_value = fake_pil_image()
    worker.model = mock_model
    return mock_model


# ───── AC1: env var honored at worker init ─────


class TestEnvVarHonoredAtInit:
    """AC1: ``SIDEQUEST_DAEMON_FIDELITY`` is the boundary that selects the
    loaded model. The worker reads it at construction; ``MODEL_VARIANT`` is
    therefore an instance attribute (not a class constant) post-45-39.
    """

    def test_default_fidelity_is_turbo_when_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In-session live narration omits the env var → must stay on Turbo
        for latency. Default is explicit, not silently inferred."""
        monkeypatch.delenv("SIDEQUEST_DAEMON_FIDELITY", raising=False)
        worker = ZImageMLXWorker(output_dir=tmp_path)
        assert worker.fidelity == "turbo"
        assert worker.model_variant == "z-image-turbo"

    def test_high_fidelity_env_var_loads_base_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        assert worker.fidelity == "high_fidelity"
        assert worker.model_variant == "z-image"

    def test_turbo_env_var_explicit_loads_turbo_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``turbo`` matches the default — still no fallback."""
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "turbo")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        assert worker.fidelity == "turbo"
        assert worker.model_variant == "z-image-turbo"

    def test_unknown_fidelity_env_var_raises_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No silent fallbacks (CLAUDE.md). A typo in the env var must fail
        at worker construction — not silently coerce to ``turbo``.
        """
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "ultra")
        with pytest.raises(ValueError, match="SIDEQUEST_DAEMON_FIDELITY"):
            ZImageMLXWorker(output_dir=tmp_path)


# ───── AC2: HF mode uses the high-fidelity tier table ─────


class TestHighFidelityTierConfig:
    """AC2: when fidelity=high_fidelity, every render pulls the 20-step,
    CFG-4.0, base-``z-image`` parameters from
    ``ZIMAGE_HIGH_FIDELITY_TIER_CONFIGS``.
    """

    def test_hf_portrait_uses_20_steps_cfg4_1024sq(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = _attached_mock_model(worker)

        worker.render(
            {"tier": "portrait", "positive_prompt": "a face", "seed": 1}
        )

        kwargs = mock_model.generate_image.call_args.kwargs
        assert kwargs["num_inference_steps"] == 20, (
            "AC2: HF portrait must use 20 steps (base 1.0)"
        )
        assert kwargs["guidance"] == 4.0, (
            "AC2: HF portrait must drive CFG 4.0 — float, not None"
        )
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 1024

    def test_hf_landscape_uses_20_steps_cfg4_landscape_aspect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = _attached_mock_model(worker)

        worker.render(
            {"tier": "landscape", "positive_prompt": "a vista", "seed": 2}
        )

        kwargs = mock_model.generate_image.call_args.kwargs
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["guidance"] == 4.0
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 768

    def test_hf_passes_guidance_float_not_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Turbo passes ``guidance=None`` (its mflux ModelConfig sets
        ``supports_guidance=False``). Base 1.0 supports CFG, so the worker
        must pass the actual float — passing None would silently disable
        CFG and produce turbo-flat output."""
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = _attached_mock_model(worker)

        worker.render(
            {"tier": "scene_illustration", "positive_prompt": "x", "seed": 0}
        )

        kwargs = mock_model.generate_image.call_args.kwargs
        assert kwargs["guidance"] is not None, (
            "AC2: HF must pass guidance float; None bypasses CFG on base 1.0"
        )
        assert kwargs["guidance"] == 4.0


# ───── AC3: mismatched-fidelity request rejected loudly ─────


class TestFidelityMismatchRejection:
    """AC3: a render request whose ``fidelity`` field does not match the
    daemon's loaded fidelity must be rejected with a structured error.

    The loaded fidelity is fixed at process start (the model is loaded
    once); a different-fidelity request cannot be satisfied without
    silently falling onto the wrong model. Per CLAUDE.md "No Silent
    Fallbacks" the daemon must fail loud.
    """

    def test_turbo_worker_rejects_high_fidelity_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SIDEQUEST_DAEMON_FIDELITY", raising=False)
        worker = ZImageMLXWorker(output_dir=tmp_path)
        _attached_mock_model(worker)

        with pytest.raises(ValueError, match="fidelity"):
            worker.render(
                {
                    "tier": "portrait",
                    "positive_prompt": "x",
                    "seed": 0,
                    "fidelity": "high_fidelity",
                }
            )

    def test_high_fidelity_worker_rejects_turbo_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        _attached_mock_model(worker)

        with pytest.raises(ValueError, match="fidelity"):
            worker.render(
                {
                    "tier": "portrait",
                    "positive_prompt": "x",
                    "seed": 0,
                    "fidelity": "turbo",
                }
            )

    def test_mismatch_error_names_both_loaded_and_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error message must surface both sides of the mismatch — the
        operator needs to know whether to relaunch the daemon or fix the
        caller. A bare ``"bad fidelity"`` message would mask which side is
        wrong.
        """
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        _attached_mock_model(worker)

        with pytest.raises(ValueError) as excinfo:
            worker.render(
                {
                    "tier": "portrait",
                    "positive_prompt": "x",
                    "seed": 0,
                    "fidelity": "turbo",
                }
            )

        msg = str(excinfo.value)
        assert "high_fidelity" in msg, (
            "AC3: mismatch error must name the loaded fidelity"
        )
        assert "turbo" in msg, (
            "AC3: mismatch error must name the requested fidelity"
        )

    def test_request_omitting_fidelity_uses_loaded_fidelity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A render request without an explicit ``fidelity`` field is
        treated as "use whatever the daemon was launched with" — common
        for legacy callers that predate the fidelity flag. Tests both
        directions to confirm no rejection on the omit path.
        """
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = _attached_mock_model(worker)

        # Should not raise.
        worker.render(
            {"tier": "portrait", "positive_prompt": "x", "seed": 0}
        )
        kwargs = mock_model.generate_image.call_args.kwargs
        assert kwargs["num_inference_steps"] == 20


# ───── AC3 (regression): turbo path unchanged ─────


class TestTurboPathUnchanged:
    """Regression guard: 45-39 must not move turbo's behavior. In-session
    live narration latency is non-negotiable.
    """

    def test_turbo_worker_still_uses_8_steps_no_guidance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SIDEQUEST_DAEMON_FIDELITY", raising=False)
        worker = ZImageMLXWorker(output_dir=tmp_path)
        mock_model = _attached_mock_model(worker)

        worker.render(
            {"tier": "portrait", "positive_prompt": "x", "seed": 0}
        )

        kwargs = mock_model.generate_image.call_args.kwargs
        assert kwargs["num_inference_steps"] == 8
        assert kwargs["guidance"] is None, (
            "Turbo must still pass guidance=None — distilled model has "
            "supports_guidance=False"
        )


# ───── AC4: OTEL spans reflect the *loaded* variant ─────


class TestOtelLoadedVariant:
    """AC4: the worker emits ``model.variant`` reflecting what it *actually
    loaded*, not what was requested. The composer separately emits
    ``model_variant`` reflecting what was *requested*. When the GM panel
    sees the two diverge, that's the diagnostic signal that
    ``SIDEQUEST_DAEMON_FIDELITY`` was set wrong at daemon launch.
    """

    def test_render_span_emits_loaded_variant_for_turbo(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        monkeypatch.delenv("SIDEQUEST_DAEMON_FIDELITY", raising=False)
        worker = ZImageMLXWorker(output_dir=tmp_path)
        _attached_mock_model(worker)

        worker.render(
            {"tier": "portrait", "positive_prompt": "x", "seed": 0}
        )

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert render_spans, "render did not emit zimage_mlx.render span"
        assert render_spans[0].attributes["model.variant"] == "z-image-turbo"

    def test_render_span_emits_loaded_variant_for_high_fidelity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        _attached_mock_model(worker)

        worker.render(
            {"tier": "portrait", "positive_prompt": "x", "seed": 0}
        )

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert render_spans, "render did not emit zimage_mlx.render span"
        assert render_spans[0].attributes["model.variant"] == "z-image", (
            "AC4: HF render span must surface the base 'z-image' variant; "
            "composer-vs-worker divergence is the misconfig diagnostic"
        )

    def test_render_span_emits_loaded_steps_for_high_fidelity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        """AC4 (cross-check with AC5 of 45-38): the ``render.steps`` span
        attribute on the worker must match the loaded tier table — 20 for
        HF, 8 for turbo. A divergence here AFTER the model_variant lines up
        would mean the env var was honored but the tier dispatch was not.
        """
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        _attached_mock_model(worker)

        worker.render(
            {"tier": "scene_illustration", "positive_prompt": "x", "seed": 0}
        )

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert render_spans
        assert render_spans[0].attributes["render.steps"] == 20

    def test_render_span_emits_worker_fidelity_attribute(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        """The render span must also expose the loaded fidelity directly so
        the GM panel doesn't have to map ``model.variant`` strings back to
        fidelity values when filtering.
        """
        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")
        worker = ZImageMLXWorker(output_dir=tmp_path)
        _attached_mock_model(worker)

        worker.render(
            {"tier": "portrait", "positive_prompt": "x", "seed": 0}
        )

        spans = otel_exporter.get_finished_spans()
        render_spans = [s for s in spans if s.name == "zimage_mlx.render"]
        assert render_spans
        assert (
            render_spans[0].attributes.get("worker.fidelity") == "high_fidelity"
        ), (
            "AC4: render span must surface worker.fidelity for GM panel filtering"
        )


# ───── AC1 wiring: WorkerPool constructs worker with env var honored ─────


class TestWorkerPoolWiring:
    """Wiring proof (CLAUDE.md "Every Test Suite Needs a Wiring Test"): the
    env-var path is reachable via the production caller. Without this, a
    future refactor could silently bypass ``__init__``'s env read by
    constructing a worker through some other path.

    ``WorkerPool.warm_up_image()`` is the canonical caller per
    test_zimage_mlx_worker.py::test_load_model_only_called_by_workerpool.
    """

    def test_warm_up_image_propagates_env_var(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sidequest_daemon.media.daemon import WorkerPool

        monkeypatch.setenv("SIDEQUEST_DAEMON_FIDELITY", "high_fidelity")

        # Patch out the heavy mflux load so this stays a unit test.
        def fake_load_model(self) -> None:
            self.model = MagicMock(name="ZImageStub")
            self.model.generate_image.return_value = fake_pil_image()

        monkeypatch.setattr(ZImageMLXWorker, "load_model", fake_load_model)
        # Skip the actual warm-up generate_image call too.
        monkeypatch.setattr(
            ZImageMLXWorker, "warm_up", lambda self: {"warmup_ms": 0}
        )

        pool = WorkerPool(output_dir=tmp_path)
        pool.warm_up_image()

        assert pool._image is not None
        assert pool._image.fidelity == "high_fidelity"
        assert pool._image.model_variant == "z-image"
