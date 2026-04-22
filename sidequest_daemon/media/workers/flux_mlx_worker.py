"""Flux MLX image generation worker — Apple Silicon native via mflux.

Replaces FluxWorker (PyTorch/diffusers) with mflux for unified memory,
zero-copy inference on M-series chips. Same interface contract:
load_model(), warm_up(), render(), cleanup().

Communicates via JSON-line protocol over stdin/stdout when run as subprocess.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path

from opentelemetry import trace

log = logging.getLogger(__name__)


# ── ADR-083 Decision 3 Layer B: matched-key visibility ─────────────────
# Cache for FluxLoRAMapping.get_mapping() — the mapping is static across
# the daemon's lifetime, so we validate its shape once on first use and
# memoize the flattened pattern list. Lazy so module import stays cheap.
_cached_lora_patterns: list[str] | None = None


def _validate_and_flatten_lora_patterns() -> list[str]:
    """Pull mflux's FluxLoRAMapping and flatten its pattern lists.

    Fails loudly if mflux's API has drifted from what Task 4.2b assumed:
    a list[LoRATarget] where each target exposes possible_{up,down,alpha}_patterns
    as iterable strings. A drift here means the matched-key count would be
    silently wrong — exactly the silent-fallback this instrumentation
    exists to surface — so we crash on first call instead of degrading.
    """
    from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
    from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping

    targets = FluxLoRAMapping.get_mapping()
    if not isinstance(targets, list):
        raise RuntimeError(
            f"mflux API drift: FluxLoRAMapping.get_mapping() returned {type(targets).__name__}, "
            f"expected list — Task 4.2b matched-key counting cannot proceed."
        )
    if not targets:
        raise RuntimeError(
            "mflux API drift: FluxLoRAMapping.get_mapping() returned an empty list — "
            "matched-key counts would all be zero."
        )

    patterns: list[str] = []
    for i, target in enumerate(targets):
        if not isinstance(target, LoRATarget):
            raise RuntimeError(
                f"mflux API drift: target[{i}] is {type(target).__name__}, expected LoRATarget."
            )
        for attr in ("possible_up_patterns", "possible_down_patterns", "possible_alpha_patterns"):
            if not hasattr(target, attr):
                raise RuntimeError(
                    f"mflux API drift: LoRATarget at index {i} missing {attr}."
                )
            patterns.extend(getattr(target, attr))
    return patterns


def _get_validated_lora_patterns() -> list[str]:
    """Memoized accessor — validates once, reuses for the daemon's lifetime."""
    global _cached_lora_patterns
    if _cached_lora_patterns is None:
        _cached_lora_patterns = _validate_and_flatten_lora_patterns()
    return _cached_lora_patterns


def _count_matched_keys_for_file(path: str, patterns: list[str]) -> int:
    """Count how many keys in a LoRA safetensors file match any mflux pattern.

    Mirrors mflux's LoRALoader._match_pattern: a {block} placeholder in a
    pattern resolves against numbers found in the weight key. A key counts
    once even if multiple patterns match (some keys map to several targets,
    e.g. fused-QKV; the count we want is "keys mflux would touch", not
    "patterns matched"). Unreadable files return 0 — matches mflux's own
    behaviour, since the render itself will fail loudly on the same path.
    """
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
    except Exception as exc:
        # safetensors_rust.SafetensorError isn't an OSError/ValueError subclass,
        # so we have to catch broadly. The render itself will fail-loudly on
        # the same unreadable path; this just prevents instrumentation from
        # masking the real error with a cryptic counter exception.
        log.warning("matched-key count: cannot open LoRA %s: %s", path, exc)
        return 0

    matched = 0
    for key in keys:
        numbers_in_key = re.findall(r"\d+", key)
        for pattern in patterns:
            if "{block}" in pattern:
                hit = False
                for num_str in numbers_in_key:
                    if key == pattern.replace("{block}", num_str):
                        hit = True
                        break
                if hit:
                    matched += 1
                    break
            elif key == pattern:
                matched += 1
                break
    return matched


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
        "portrait_square": {
            "model": "dev",
            "steps": 12,
            "guidance": 3.5,
            "w": 1024,
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
            "model": "dev",
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

    def load_model(self, variant: str = "dev") -> None:
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
        """MLX graph compilation via dev dummy generation."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.flux_mlx_worker")
        with tracer.start_as_current_span("flux_mlx.warm_up") as span:
            start = time.monotonic()

            self.models["dev"].generate_image(
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

    def _count_matched_keys(self, lora_paths: list[str]) -> list[int]:
        """Per-file count of LoRA keys mflux's loader will recognise.

        ADR-083 Decision 3 Layer B: a stale, mis-trained, or wrong-flavour
        LoRA can survive `608/608 keys matched` from the loader yet
        contribute almost nothing to the render. Surfacing this count on
        every render lets the GM panel spot adapters that are silently
        no-ops without waiting for a human to notice "that doesn't look
        right". Returns one int per input path, in order.
        """
        patterns = _get_validated_lora_patterns()
        return [_count_matched_keys_for_file(p, patterns) for p in lora_paths]

    def _build_lora_model(
        self, variant: str, lora_paths: list[str], lora_scales: list[float]
    ) -> object:
        """Construct a Flux1 instance with one or more LoRA adapters applied.

        Uses Flux1(model_config=..., lora_paths=..., lora_scales=...) because
        Flux1.from_name() does not accept LoRA parameters.

        Per ADR-083 (Decision 4), the protocol is array-only. Callers pass
        lists even for single-LoRA renders — no singleton compat shim.
        """
        from mflux.models.flux.variants.txt2img.flux import Flux1
        from mflux.models.common.config.model_config import ModelConfig

        config_factory = {"dev": ModelConfig.dev, "schnell": ModelConfig.schnell}
        if variant not in config_factory:
            raise ValueError(f"Unknown variant for LoRA: {variant!r}")
        if len(lora_paths) != len(lora_scales):
            raise ValueError(
                f"lora_paths ({len(lora_paths)}) / lora_scales ({len(lora_scales)}) "
                f"length mismatch"
            )

        return Flux1(
            model_config=config_factory[variant](),
            quantize=self.QUANTIZE,
            lora_paths=list(lora_paths),
            lora_scales=list(lora_scales),
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
                # Story 35-15: variant override from the genre pack's
                # `visual_style.yaml::preferred_model`. Empty string (or
                # missing) → fall back to the tier's default variant. Any
                # non-empty value MUST be a known variant — no silent
                # fallback, unknown values raise loudly so a misconfigured
                # genre pack fails the render instead of silently
                # downgrading to the tier default.
                requested_variant = params.get("variant", "") or ""
                if requested_variant:
                    if requested_variant not in {"dev", "schnell"}:
                        raise ValueError(
                            f"Unknown variant override {requested_variant!r} "
                            f"for tier {tier_name!r}. Valid values: 'dev', 'schnell'. "
                            f"Check visual_style.yaml::preferred_model."
                        )
                    variant = requested_variant
                else:
                    variant = tier_cfg["model"]

                prompt = self._compose_prompt(params)
                seed = params.get("seed", 0)
                # Per ADR-083 Decision 4: protocol is array-only. Legacy
                # singleton params (lora_path/lora_scale) are no longer
                # accepted — no silent fallback. Caller must provide arrays.
                if "lora_path" in params or "lora_scale" in params:
                    raise ValueError(
                        "legacy singleton params lora_path/lora_scale are no longer "
                        "accepted — use lora_paths[] and lora_scales[] arrays."
                    )
                lora_paths: list[str] = list(params.get("lora_paths") or [])
                lora_scales: list[float] = list(params.get("lora_scales") or [])

                span.set_attribute("render.tier", tier_name)
                span.set_attribute("render.seed", seed)
                span.set_attribute("render.variant", variant)
                span.set_attribute("render.width", tier_cfg["w"])
                span.set_attribute("render.height", tier_cfg["h"])
                # LoRA attributes attach to the existing flux_mlx.render span
                # per ADR-083 Decision 3 (Architect correction #1) — one span
                # per render, not a separate render.lora span.
                span.set_attribute("render.lora.stack_size", len(lora_paths))

                if lora_paths:
                    span.set_attribute("render.lora.files", lora_paths)
                    span.set_attribute("render.lora.scales", lora_scales)
                    # ADR-083 Decision 3 Layer B (Task 4.2b): per-file count
                    # of how many LoRA keys mflux's mapping recognises. A
                    # zero or near-zero entry here means that adapter is
                    # effectively a no-op even if loading "succeeded".
                    matched_keys = self._count_matched_keys(lora_paths)
                    span.set_attribute("render.lora.matched_keys", matched_keys)
                    log.info(
                        "FLUX MLX RENDER [%s] seed=%s loras=%s scales=%s matched=%s",
                        tier_name, seed, lora_paths, lora_scales, matched_keys,
                    )
                    model = self._build_lora_model(variant, lora_paths, lora_scales)
                else:
                    log.info("FLUX MLX RENDER [%s] seed=%s (no LoRA)", tier_name, seed)
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
