"""Flux MLX image generation worker — Apple Silicon native via mflux.

Replaces FluxWorker (PyTorch/diffusers) with mflux for unified memory,
zero-copy inference on M-series chips. Same interface contract:
load_model(), warm_up(), render(), cleanup().

Communicates via JSON-line protocol over stdin/stdout when run as subprocess.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path

from opentelemetry import trace

log = logging.getLogger(__name__)


class FluxMLXWorker:
    """Flux image generation worker using Apple MLX via mflux."""

    # Tier config — KEEP IN SYNC with flux_config.py and daemon.py FLUX_TIERS.
    TIER_CONFIGS = {
        "scene_illustration": {
            "model": "dev",
            "steps": 12,
            "guidance": 3.5,
            "w": 1024,
            "h": 768,
        },
        "portrait": {
            "model": "dev",
            "steps": 12,
            "guidance": 3.5,
            "w": 768,
            "h": 1024,
        },
        "landscape": {
            "model": "dev",
            "steps": 12,
            "guidance": 3.5,
            "w": 1024,
            "h": 768,
        },
        "text_overlay": {
            "model": "schnell",
            "steps": 4,
            "guidance": 0.0,
            "w": 768,
            "h": 512,
        },
        "cartography": {
            "model": "dev",
            "steps": 20,
            "guidance": 3.5,
            "w": 1024,
            "h": 1024,
        },
        "tactical_sketch": {
            "model": "dev",
            "steps": 12,
            "guidance": 3.5,
            "w": 1024,
            "h": 1024,
        },
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict = {}
        self._active_variant: str | None = None

    # Quantization level for model loading. None = full precision.
    # Runtime quantization (4/8) adds overhead on mflux 0.17.4 without speedup.
    # Re-evaluate when pre-quantized checkpoints are available.
    QUANTIZE: int | None = None

    def load_model(self, variant: str = "schnell") -> None:
        """Load a Flux model variant via mflux (MLX native)."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.flux_mlx_worker")
        with tracer.start_as_current_span("flux_mlx.load_model") as span:
            span.set_attribute("model.variant", variant)
            span.set_attribute("model.quantize", self.QUANTIZE or 0)
            from mflux.models.flux.variants.txt2img.flux import Flux1

            self.models[variant] = Flux1.from_name(
                model_name=variant, quantize=self.QUANTIZE
            )
            self._active_variant = variant

    def _ensure_variant(self, variant: str) -> None:
        """Lazy-load a variant if not already loaded."""
        if variant not in self.models:
            self.load_model(variant)

    def warm_up(self) -> dict:
        """MLX graph compilation via schnell dummy generation."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.flux_mlx_worker")
        with tracer.start_as_current_span("flux_mlx.warm_up") as span:
            start = time.monotonic()

            self.models["schnell"].generate_image(
                prompt="black",
                num_inference_steps=1,
                guidance=0.0,
                width=512,
                height=512,
                seed=0,
            )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            span.set_attribute("warmup.elapsed_ms", elapsed_ms)
            return {"warmup_ms": elapsed_ms}

    def _build_lora_model(self, variant: str, lora_path: str, lora_scale: float) -> object:
        """Construct a Flux1 instance with LoRA weights.

        Uses Flux1(model_config=..., lora_paths=..., lora_scales=...) because
        Flux1.from_name() does not accept LoRA parameters.
        """
        from mflux.models.flux.variants.txt2img.flux import Flux1
        from mflux.models.common.config.model_config import ModelConfig

        config_factory = {"dev": ModelConfig.dev, "schnell": ModelConfig.schnell}
        if variant not in config_factory:
            raise ValueError(f"Unknown variant for LoRA: {variant!r}")

        return Flux1(
            model_config=config_factory[variant](),
            quantize=self.QUANTIZE,
            lora_paths=[lora_path],
            lora_scales=[lora_scale],
        )

    def render(self, params: dict) -> dict:
        """Generate image from StageCue params. Returns result dict."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.flux_mlx_worker")
        with tracer.start_as_current_span("flux_mlx.render") as span:
            try:
                tier_name = params.get("tier", "")
                if tier_name not in self.TIER_CONFIGS:
                    raise ValueError(f"Unsupported tier: {tier_name!r}")

                tier_cfg = self.TIER_CONFIGS[tier_name]
                variant = tier_cfg["model"]

                prompt = self._compose_prompt(params)
                seed = params.get("seed", 0)
                lora_path = params.get("lora_path")
                lora_scale = params.get("lora_scale", 1.0)

                span.set_attribute("render.tier", tier_name)
                span.set_attribute("render.seed", seed)
                span.set_attribute("render.variant", variant)
                span.set_attribute("render.width", tier_cfg["w"])
                span.set_attribute("render.height", tier_cfg["h"])

                if lora_path:
                    span.set_attribute("render.lora_path", lora_path)
                    span.set_attribute("render.lora_scale", lora_scale)
                    log.info(
                        "FLUX MLX RENDER [%s] seed=%s lora=%s scale=%s",
                        tier_name, seed, lora_path, lora_scale,
                    )
                    model = self._build_lora_model(variant, lora_path, lora_scale)
                else:
                    log.info("FLUX MLX RENDER [%s] seed=%s", tier_name, seed)
                    self._ensure_variant(variant)
                    model = self.models[variant]

                log.info("  prompt: %s", prompt[:150])

                start = time.monotonic()
                image = model.generate_image(
                    prompt=prompt,
                    num_inference_steps=tier_cfg["steps"],
                    guidance=tier_cfg["guidance"],
                    width=tier_cfg["w"],
                    height=tier_cfg["h"],
                    seed=seed,
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)
                span.set_attribute("render.elapsed_ms", elapsed_ms)

                filename = f"render_{uuid.uuid4().hex[:8]}.png"
                image_path = self.output_dir / filename
                image.save(str(image_path))

                return {
                    "image_url": str(image_path),
                    "width": tier_cfg["w"],
                    "height": tier_cfg["h"],
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    def _compose_prompt(self, params: dict) -> str:
        """Build positive prompt for Flux.

        If SubprocessRenderer already composed a prompt (with genre style
        suffix and location tag overrides), use it directly. Otherwise
        fall back to building from raw StageCue fields.
        """
        # Prefer pre-composed prompt from SubprocessRenderer/PromptComposer
        if params.get("positive_prompt"):
            return params["positive_prompt"]

        # Accept raw prompt from batch scripts (generate_portraits.py, etc.)
        if params.get("prompt"):
            return params["prompt"]

        tier = params.get("tier", "")
        is_text_overlay = tier == "text_overlay"

        parts: list[str] = []
        if params.get("subject"):
            subject = params["subject"]
            if is_text_overlay:
                subject = f"text reading {subject}"
            parts.append(subject)
        if params.get("mood"):
            parts.append(f"{params['mood']} atmosphere")
        if params.get("location"):
            parts.append(f"set in {params['location']}")
        if params.get("tags"):
            parts.extend(params["tags"])

        if is_text_overlay:
            parts.extend(["clean typography", "readable text", "sharp lettering"])

        if not parts:
            raise ValueError(
                "No prompt content: params has no positive_prompt, prompt, subject, or tags. "
                f"Params: {params}"
            )

        return ", ".join(parts)

    def cleanup(self) -> None:
        """Unload models, free GPU memory."""
        self.models.clear()
        self._active_variant = None

def _respond(
    req_id: str, *, result: dict | None = None, error: dict | None = None
) -> None:
    """Write a JSON response line to stdout."""
    resp: dict = {"id": req_id}
    if result is not None:
        resp["result"] = result
    if error is not None:
        resp["error"] = error
    print(json.dumps(resp), flush=True)


def main() -> None:
    """JSON-line protocol loop."""
    import tempfile

    output_dir = Path(tempfile.mkdtemp(prefix="sq-flux-mlx-"))
    worker = FluxMLXWorker(output_dir)

    worker.load_model()
    worker.warm_up()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            req_id = req.get("id", "unknown")
            method = req.get("method")
            if not method:
                _respond(
                    req_id,
                    error={"code": "INVALID_REQUEST", "message": "Missing 'method'"},
                )
                continue
        except json.JSONDecodeError as e:
            _respond("unknown", error={"code": "PARSE_ERROR", "message": str(e)})
            continue
        params = req.get("params", {})

        if method == "ping":
            _respond(req_id, result={"status": "ok"})
        elif method == "shutdown":
            _respond(req_id, result={"status": "ok"})
            worker.cleanup()
            break
        elif method == "render":
            try:
                render_result = worker.render(params)
                _respond(req_id, result=render_result)
            except Exception as e:
                _respond(req_id, error={"code": "GENERATION_FAILED", "message": str(e)})
        else:
            _respond(
                req_id,
                error={"code": "UNKNOWN_METHOD", "message": f"Unknown: {method}"},
            )


if __name__ == "__main__":
    main()
