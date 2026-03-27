"""Flux image generation worker — runs as isolated subprocess.

Communicates via JSON-line protocol over stdin/stdout.
Designed to run in a separate venv (~/.venvs/flux/) with torch/diffusers.
Does not import from sidequest at runtime.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Suppress CLIP token truncation warnings — T5 handles full 512 tokens
logging.getLogger("transformers").setLevel(logging.ERROR)


class FluxWorker:
    """Flux image generation worker — dev for illustrations, schnell for text."""

    # Tier config duplicated from flux_config.py (worker cannot import sidequest).
    # KEEP IN SYNC with flux_config.py — worker runs in isolated subprocess.
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

    MODEL_IDS = {
        "schnell": "black-forest-labs/FLUX.1-schnell",
        "dev": "black-forest-labs/FLUX.1-dev",
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipes: dict = {}
        self._active_variant: str | None = None

    def load_model(self, variant: str = "schnell") -> None:
        """Load a Flux model variant to MPS in float16."""
        import torch
        from diffusers import FluxPipeline

        model_id = self.MODEL_IDS[variant]
        self.pipes[variant] = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("mps")
        self._active_variant = variant

    def _ensure_variant(self, variant: str) -> None:
        """Lazy-load a variant if not already loaded."""
        if variant not in self.pipes:
            self.load_model(variant)

    def warm_up(self) -> dict:
        """MPS graph compilation via schnell dummy generation."""
        import torch

        start = time.monotonic()
        generator = torch.Generator("mps").manual_seed(0)

        self.pipes["schnell"](
            prompt="black",
            num_inference_steps=1,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=generator,
        )
        torch.mps.empty_cache()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {"warmup_ms": elapsed_ms}

    def render(self, params: dict) -> dict:
        """Generate image from StageCue params. Returns result dict."""
        import torch

        tier_name = params.get("tier", "")
        if tier_name not in self.TIER_CONFIGS:
            raise ValueError(f"Unsupported tier: {tier_name!r}")

        tier_cfg = self.TIER_CONFIGS[tier_name]
        variant = tier_cfg["model"]
        self._ensure_variant(variant)

        prompt = self._compose_prompt(params)
        clip_prompt = params.get("clip_prompt", "")

        seed = params.get("seed", 0)
        generator = torch.Generator("mps").manual_seed(seed)

        # Flux dual encoder: prompt → CLIP (style), prompt_2 → T5 (content).
        # When clip_prompt is provided, route style keywords to CLIP and full
        # content to T5. Otherwise, single prompt goes to both encoders.
        pipe_kwargs: dict = {
            "num_inference_steps": tier_cfg["steps"],
            "guidance_scale": tier_cfg["guidance"],
            "width": tier_cfg["w"],
            "height": tier_cfg["h"],
            "generator": generator,
            "max_sequence_length": 512,
        }
        if clip_prompt:
            pipe_kwargs["prompt"] = clip_prompt
            pipe_kwargs["prompt_2"] = prompt
        else:
            pipe_kwargs["prompt"] = prompt

        start = time.monotonic()
        result = self.pipes[variant](**pipe_kwargs)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        image = result.images[0]
        filename = f"render_{uuid.uuid4().hex[:8]}.png"
        image_path = self.output_dir / filename
        image.save(str(image_path))

        torch.mps.empty_cache()

        return {
            "image_url": str(image_path),
            "width": tier_cfg["w"],
            "height": tier_cfg["h"],
            "elapsed_ms": elapsed_ms,
        }

    def _compose_prompt(self, params: dict) -> str:
        """Build positive prompt for Flux.

        If SubprocessRenderer already composed a prompt (with genre style
        suffix and location tag overrides), use it directly. Otherwise
        fall back to building from raw StageCue fields.
        """
        # Prefer pre-composed prompt from SubprocessRenderer/PromptComposer
        if params.get("positive_prompt"):
            return params["positive_prompt"]

        tier = params.get("tier", "")
        is_text_overlay = tier == "text_overlay"

        parts = []
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

        return ", ".join(parts) if parts else "fantasy scene"

    def cleanup(self) -> None:
        """Release all loaded models and clear MPS cache."""
        import torch

        self.pipes.clear()
        self._active_variant = None
        torch.mps.empty_cache()


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
    output_dir = Path(tempfile.mkdtemp(prefix="sq-flux-"))
    worker = FluxWorker(output_dir)

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
