"""Z-Image image generation worker — Apple Silicon native via mflux.

Replaces FluxMLXWorker. Same interface contract: load_model(), warm_up(),
render(), cleanup(). Communicates via JSON-line protocol over stdin/stdout
when run as subprocess. No LoRA support.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from opentelemetry import trace

from sidequest_daemon.media.camera_specs import CameraLoader
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    CharacterTokens,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.prompt_composer import PromptComposer
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import (
    LOD,
    CameraPreset,
    ComposedPrompt,
    RenderConfigError,
    RenderTarget,
)
from sidequest_daemon.renderer.models import RenderTier, StageCue

log = logging.getLogger(__name__)

_DAEMON_ROOT = Path(__file__).resolve().parents[3]


def _get_composer(
    genre: str,
    world: str,
    *,
    pc_descriptor: dict | None = None,
) -> PromptComposer:
    """Build a composer scoped to the target's world.

    When ``pc_descriptor`` is supplied (slice 2 of catalog-injected compose
    wiring), the PC is registered into the freshly-loaded ``CharacterCatalog``
    via ``add_pc`` before the composer is returned. The descriptor lets the
    server hand a ``pc:<slug>`` ref to the daemon without first persisting a
    ``portrait_manifest`` entry — a single appearance prose is replicated to
    every ``LOD`` (mirror of ``CharacterCatalog.load``'s production path so
    PCs and NPCs share the same eviction ladder).
    """
    packs_root = Path(os.environ["SIDEQUEST_GENRE_PACKS"])
    characters = CharacterCatalog.load(packs_root, genre=genre, world=world)
    if pc_descriptor is not None:
        pc_id = pc_descriptor["id"]
        appearance = pc_descriptor.get("appearance", "")
        tokens = CharacterTokens(
            kind="pc",
            descriptions=dict.fromkeys(LOD, appearance),
            default_pose=pc_descriptor.get("default_pose", ""),
            culture=pc_descriptor.get("culture"),
            world=world,
        )
        characters.add_pc(pc_id, tokens)
    return PromptComposer(
        recipes=RecipeLoader.from_file(_DAEMON_ROOT / "recipes.yaml"),
        cameras=CameraLoader.from_file(_DAEMON_ROOT / "cameras.yaml"),
        characters=characters,
        places=PlaceCatalog.load(packs_root, genre=genre, world=world),
        styles=StyleCatalog.load(packs_root, genre=genre, world=world),
    )


def build_render_target(cue: StageCue) -> RenderTarget:
    """Translate a StageCue into a RenderTarget.

    `cue.metadata["world"]` and `cue.metadata["genre"]` are required — fail
    loud if either is missing.
    """
    world = cue.metadata.get("world")
    genre = cue.metadata.get("genre")
    if not world or not genre:
        raise ValueError(
            "StageCue.metadata must carry `world` and `genre` for composer routing",
        )

    if cue.tier in (RenderTier.PORTRAIT, RenderTier.PORTRAIT_SQUARE):
        character = cue.characters[0] if cue.characters else cue.subject
        return RenderTarget(
            kind="portrait",
            world=world,
            genre=genre,
            character=character,
            camera=cue.camera,
        )
    if cue.tier == RenderTier.LANDSCAPE:
        # POI render: subject is a `where:` ref.
        return RenderTarget(
            kind="poi",
            world=world,
            genre=genre,
            place=cue.subject,
        )
    if cue.tier == RenderTier.SCENE_ILLUSTRATION:
        return RenderTarget(
            kind="illustration",
            world=world,
            genre=genre,
            participants=cue.characters,
            location=cue.location or cue.metadata.get("location_ref", ""),
            action=cue.subject,
            camera=cue.camera or CameraPreset.scene,
        )
    raise ValueError(f"unsupported tier for composer routing: {cue.tier!r}")


def build_cue_from_params(params: dict) -> StageCue:
    """Project a daemon `render` request's params dict into a ``StageCue``.

    Pulled out of the `_handle_client` dispatch loop so the params→metadata
    projection (specifically: forwarding ``pc_descriptor`` for slice 2 of the
    catalog-injected compose wiring) can be tested without a live socket. The
    dispatch loop's only responsibility on top of this helper is the
    early-out conditional checking that subject/world/genre are present.
    """
    tier_str = params.get("tier", "scene_illustration")
    tier = (
        RenderTier(tier_str)
        if tier_str in {t.value for t in RenderTier}
        else RenderTier.SCENE_ILLUSTRATION
    )
    metadata: dict = {
        "world": params["world"],
        "genre": params["genre"],
    }
    pc_descriptor = params.get("pc_descriptor")
    if pc_descriptor is not None:
        metadata["pc_descriptor"] = pc_descriptor
    return StageCue(
        subject=params.get("subject", ""),
        tier=tier,
        location=params.get("location", ""),
        mood=params.get("mood", ""),
        characters=params.get("characters", []),
        tags=params.get("tags", []),
        metadata=metadata,
    )


def compose_prompt_for(cue: StageCue) -> ComposedPrompt:
    """Build a ComposedPrompt from a StageCue end-to-end.

    When ``cue.metadata['pc_descriptor']`` is set, the PC is registered into
    the catalog before composition runs — the server's ``pc:<slug>`` ref will
    resolve without a disk lookup.
    """
    world = cue.metadata["world"]
    genre = cue.metadata["genre"]
    pc_descriptor = cue.metadata.get("pc_descriptor")
    composer = _get_composer(genre, world, pc_descriptor=pc_descriptor)
    target = build_render_target(cue)
    return composer.compose(target)




class ZImageMLXWorker:
    """Z-Image Turbo image generation worker using Apple MLX via mflux.

    Migrated from `z-image` (base, 20 steps, CFG 4.0) to `z-image-turbo`
    (LCM-distilled, 8 steps, no CFG) on 2026-04-26 per the S4-PERF
    investigation. Per-render wall-clock target: ~30s (was ~108s).

    **Per-process singleton invariant** (Story 43-5): only one
    `ZImageMLXWorker` may exist per Python process. A second construction
    raises ``RuntimeError`` to fail loudly per CLAUDE.md "No Silent
    Fallbacks." Z-Image survives a second model on the same MPS device,
    but Flux historically OOM'd the M3 Max instantly — the invariant is
    here to prevent any future renderer revert from silently spawning a
    second model. Production callers must route through
    ``WorkerPool.warm_up_image()`` (``sidequest_daemon/media/daemon.py``),
    which also guards via ``_image_loaded``. The conftest autouse fixture
    resets ``ZImageMLXWorker._instance = None`` between tests.
    """

    # Per-process singleton handle. Set by __init__ on first construction;
    # raises RuntimeError on second. Tests reset to None between cases.
    _instance: "ZImageMLXWorker | None" = None

    # The mflux model alias passed to ModelConfig.from_name. The string is
    # also written to OTEL spans as `model.variant` so the GM panel can
    # tell turbo and base-Z-Image renders apart at a glance.
    MODEL_VARIANT: str = "z-image-turbo"

    # Tier config — KEEP IN SYNC with zimage_config.py and daemon.py IMAGE_TIERS.
    # Turbo is distilled, so guidance is fixed at 0.0 (CFG is a no-op).
    TIER_CONFIGS = {
        "scene_illustration": {"steps": 8, "guidance": 0.0, "w": 1024, "h": 768},
        "portrait": {"steps": 8, "guidance": 0.0, "w": 768, "h": 1024},
        "portrait_square": {"steps": 8, "guidance": 0.0, "w": 1024, "h": 1024},
        "landscape": {"steps": 8, "guidance": 0.0, "w": 1024, "h": 768},
        "text_overlay": {"steps": 8, "guidance": 0.0, "w": 768, "h": 512},
        "cartography": {"steps": 8, "guidance": 0.0, "w": 1024, "h": 1024},
        "fog_of_war": {"steps": 8, "guidance": 0.0, "w": 1024, "h": 1024},
    }

    # Quantization level for model loading. 8-bit per mflux's Turbo README
    # example (`--steps 9 --quantize 8`). None = full precision.
    QUANTIZE: int | None = 8

    # Z-Image's default scheduler per its CLI.
    #
    # Draw Things delta: Draw Things validation used "UniPC Trailing", which
    # empirically produces slightly cleaner detail at 20 steps / CFG 4.0.
    # mflux 0.x does NOT ship UniPC — the only options in
    # mflux/models/common/schedulers/ are:
    #   - linear
    #   - flow_match_euler_discrete  (this one; used for Z-Image with CFG>1)
    #   - seedvr2_euler              (SeedVR2-specific)
    # If/when mflux adds UniPC this is the line to change. For now the daemon
    # is NOT a pixel-perfect match to Draw Things on sampler, only on
    # steps/guidance/resolution-dependent-shift.
    #
    # Resolution-dependent shift ("Resolution Dpt. Shift" in Draw Things) IS
    # enabled automatically: mflux's ModelConfig for "z-image" and
    # "z-image-turbo" both set requires_sigma_shift=True, which causes
    # Config.scheduler to call FlowMatchEulerDiscreteScheduler.set_image_seq_len
    # at construction time — i.e. the sigma schedule is recomputed per call
    # based on (width/16) * (height/16). No extra flag needed here.
    #
    # CFG Zero*: no such toggle exists in mflux. Nothing to disable.
    SCHEDULER: str = "flow_match_euler_discrete"

    def __init__(self, output_dir: Path) -> None:
        if type(self)._instance is not None:
            raise RuntimeError(
                "ZImageMLXWorker is a per-process singleton; a second "
                "construction would load a second model on the same MPS "
                "device. Route through WorkerPool.warm_up_image() instead, "
                "or reset ZImageMLXWorker._instance=None in tests."
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model: object | None = None
        type(self)._instance = self

    def load_model(self) -> None:
        """Load the configured Z-Image variant via mflux."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.zimage_mlx_worker")
        with tracer.start_as_current_span("zimage_mlx.load_model") as span:
            # `model.name` retains the historical "z-image" value so existing
            # OTEL queries / dashboards keep working. `model.variant` is the
            # new-since-2026-04-26 attribute that distinguishes turbo from
            # base — the GM panel filters on this.
            span.set_attribute("model.name", "z-image")
            span.set_attribute("model.variant", self.MODEL_VARIANT)
            span.set_attribute("model.quantize", self.QUANTIZE or 0)
            from mflux.models.common.config import ModelConfig
            from mflux.models.z_image.variants.z_image import ZImage

            self.model = ZImage(
                model_config=ModelConfig.from_name(self.MODEL_VARIANT),
                quantize=self.QUANTIZE,
            )

    def warm_up(self) -> dict:
        """MLX graph compilation via dummy generation.

        Caller (``WorkerPool.warm_up_image()``) is contractually required
        to invoke ``load_model()`` first. We raise rather than silently
        lazy-load — `assert` is unsafe under Python `-O` (assertions are
        stripped), so the check is an explicit `raise`.
        """
        if self.model is None:
            raise RuntimeError(
                "warm_up() called before load_model() — caller contract "
                "violation. WorkerPool.warm_up_image() loads the model "
                "before warm_up; do not call warm_up directly."
            )
        tracer = trace.get_tracer("sidequest_daemon.media.workers.zimage_mlx_worker")
        with tracer.start_as_current_span("zimage_mlx.warm_up") as span:
            start = time.monotonic()
            self.model.generate_image(  # type: ignore[attr-defined]
                seed=0,
                prompt="black",
                num_inference_steps=2,
                guidance=None,
                width=512,
                height=512,
                scheduler=self.SCHEDULER,
                negative_prompt=None,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            span.set_attribute("warmup.elapsed_ms", elapsed_ms)
            return {"warmup_ms": elapsed_ms}

    def render(self, params: dict) -> dict:
        """Generate image from StageCue params. Returns result dict."""
        tracer = trace.get_tracer("sidequest_daemon.media.workers.zimage_mlx_worker")
        with tracer.start_as_current_span("zimage_mlx.render") as span:
            try:
                tier_name = params.get("tier", "")
                if tier_name not in self.TIER_CONFIGS:
                    raise ValueError(f"Unsupported tier: {tier_name!r}")

                tier_cfg = self.TIER_CONFIGS[tier_name]
                prompt = self._compose_prompt(params)
                negative_prompt = params.get("negative_prompt") or None
                seed = params.get("seed", 0)

                span.set_attribute("model.variant", self.MODEL_VARIANT)
                span.set_attribute("render.tier", tier_name)
                span.set_attribute("render.seed", seed)
                span.set_attribute("render.width", tier_cfg["w"])
                span.set_attribute("render.height", tier_cfg["h"])
                span.set_attribute("render.steps", tier_cfg["steps"])
                span.set_attribute("render.guidance", tier_cfg["guidance"])
                span.set_attribute("render.prompt_length", len(prompt))
                span.set_attribute("render.negative_length", len(negative_prompt or ""))

                log.info(
                    "ZIMAGE RENDER [%s] seed=%s w=%s h=%s steps=%s",
                    tier_name,
                    seed,
                    tier_cfg["w"],
                    tier_cfg["h"],
                    tier_cfg["steps"],
                )
                log.info("  prompt: %s", prompt[:150])

                if self.model is None:
                    raise RuntimeError(
                        "render() called before load_model() — caller "
                        "contract violation. Route through WorkerPool which "
                        "guards via _image_loaded."
                    )

                # Z-Image Turbo's ModelConfig sets supports_guidance=False;
                # pass guidance=None (mflux's "disabled" sentinel) when the
                # tier guidance is 0.0 so we don't accidentally activate a
                # CFG path on a distilled model. Base Z-Image (guidance>0)
                # would still send the float through unchanged if reverted.
                guidance_arg: float | None = (
                    tier_cfg["guidance"] if tier_cfg["guidance"] > 0.0 else None
                )

                start = time.monotonic()
                image = self.model.generate_image(  # type: ignore[attr-defined]
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=tier_cfg["steps"],
                    guidance=guidance_arg,
                    width=tier_cfg["w"],
                    height=tier_cfg["h"],
                    scheduler=self.SCHEDULER,
                    negative_prompt=negative_prompt,
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
        if params.get("positive_prompt"):
            return params["positive_prompt"]
        if params.get("prompt"):
            return params["prompt"]
        raise RenderConfigError(
            "compose pipeline failed to produce a prompt; "
            f"params keys={sorted(params.keys())}"
        )

    def cleanup(self) -> None:
        """Unload model, free GPU memory."""
        self.model = None
        # Release the singleton slot so a fresh process or test fixture
        # can construct a new worker without tripping the singleton guard.
        type(self)._instance = None
