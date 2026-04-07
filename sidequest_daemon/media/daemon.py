"""sidequest-renderer daemon — persistent Flux image renderer on Unix domain socket.

Hosts the Flux image worker in a single process with the model pre-loaded.
Serves render requests over a Unix domain socket, routing by tier.
Stays warm between sessions.

Usage:
    sidequest-renderer                          # start daemon (loads Flux)
    sidequest-renderer --warmup=flux            # start + load Flux only
    sidequest-renderer --no-warmup              # start without loading models (testing)
    sidequest-renderer --shutdown               # send shutdown to running daemon
    sidequest-renderer --status                 # check daemon status
    sidequest-renderer --genre-packs /path      # set genre packs directory
    sidequest-renderer --output-dir /path       # set output directory
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path

SOCKET_PATH = Path("/tmp/sidequest-renderer.sock")
PID_PATH = Path("/tmp/sidequest-renderer.pid")

log = logging.getLogger(__name__)

# Tier → worker routing.
FLUX_TIERS = frozenset({"scene_illustration", "portrait", "landscape", "cartography", "text_overlay", "tactical_sketch"})
EMBED_TIERS = frozenset({"embed"})


class EmbedWorker:
    """Generates sentence embeddings via sentence-transformers (story 15-7).

    Uses all-MiniLM-L6-v2 for 384-dimensional embeddings — fast and
    good enough for lore fragment similarity search.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_name = "all-MiniLM-L6-v2"

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a sentence embedding for the given text.

        Raises ValueError if text is empty — no silent fallbacks.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return [float(v) for v in embedding]


class WorkerPool:
    """Manages the Flux image worker with lazy or eager loading."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._flux = None
        self._acestep = None
        self._flux_loaded = False
        self._acestep_loaded = False
        self.pipeline_factory = None  # Set by _run_daemon after init

        # GPU memory coordinator — manages 80GB shared budget across backends
        from sidequest_daemon.ml.memory_manager import ModelMemoryManager
        self.memory_manager = ModelMemoryManager()

    def warm_up_flux(self) -> dict:
        """Load and warm up Flux worker (both schnell and dev variants)."""
        if self._flux_loaded:
            return {"worker": "flux", "status": "already_warm", "warmup_ms": 0}
        from sidequest_daemon.media.workers.flux_worker import FluxWorker
        self._flux = FluxWorker(self.output_dir / "flux")
        log.info("Loading Flux schnell (text overlays)...")
        self._flux.load_model("schnell")
        log.info("Loading Flux dev (cartography)...")
        self._flux.load_model("dev")
        result = self._flux.warm_up()
        self._flux_loaded = True
        log.info("Flux warm — both variants (%.1fs)", result.get("warmup_ms", 0) / 1000)
        return {"worker": "flux", "status": "warm", "variants": ["schnell", "dev"], **result}

    def _ensure_flux(self) -> None:
        if not self._flux_loaded:
            self.warm_up_flux()

    def warm_up_acestep(self) -> dict:
        """Load and warm up ACE-Step music worker."""
        if self._acestep_loaded:
            return {"worker": "acestep", "status": "already_warm", "warmup_ms": 0}
        from sidequest_daemon.media.workers.acestep_worker import ACEStepWorker
        self._acestep = ACEStepWorker(self.output_dir / "acestep")
        self._acestep.load_model()
        result = self._acestep.warm_up()
        self._acestep_loaded = True
        log.info("ACE-Step warm (%.1fs)", result.get("warmup_ms", 0) / 1000)
        return {"worker": "acestep", "status": "warm", **result}

    def _ensure_acestep(self) -> None:
        if not self._acestep_loaded:
            self.warm_up_acestep()

    def render(self, params: dict) -> dict:
        """Route render request to the appropriate worker by tier."""
        tier = params.get("tier", "")
        if tier in FLUX_TIERS:
            self._ensure_flux()
            return self._flux.render(params)
        else:
            raise ValueError(f"Unknown tier: {tier!r}")

    def status(self) -> dict:
        """Return current worker status."""
        return {
            "flux": "warm" if self._flux_loaded else "cold",
            "acestep": "warm" if self._acestep_loaded else "cold",
            "supported_tiers": {
                "flux": sorted(FLUX_TIERS),
            },
        }

    def cleanup(self) -> None:
        """Release all models and clear GPU cache."""
        if self._flux is not None:
            self._flux.cleanup()
            self._flux = None
            self._flux_loaded = False
        if self._acestep is not None:
            self._acestep.cleanup()
            self._acestep = None
            self._acestep_loaded = False


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    pool: WorkerPool,
    render_lock: asyncio.Lock,
) -> None:
    """Handle a single client connection — read JSON lines, dispatch, respond."""
    peer = writer.get_extra_info("peername") or "unix-client"
    log.info("Client connected: %s", peer)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            line_str = line.decode().strip()
            if not line_str:
                continue

            try:
                req = json.loads(line_str)
                req_id = req.get("id", "unknown")
                method = req.get("method")
            except json.JSONDecodeError as e:
                _write(writer, "unknown", error={"code": "PARSE_ERROR", "message": str(e)})
                continue

            if not method:
                _write(writer, req_id, error={"code": "INVALID_REQUEST", "message": "Missing 'method'"})
                continue

            params = req.get("params", {})

            if method == "ping":
                _write(writer, req_id, result={"status": "ok"})
            elif method == "status":
                _write(writer, req_id, result=pool.status())
            elif method == "shutdown":
                _write(writer, req_id, result={"status": "ok"})
                log.info("Shutdown requested by client")
                asyncio.get_event_loop().call_soon(lambda: os.kill(os.getpid(), signal.SIGTERM))
            elif method == "warm_up":
                try:
                    target = params.get("worker", "all")
                    results = {}
                    if target in ("all", "flux"):
                        results["flux"] = await asyncio.to_thread(pool.warm_up_flux)
                    if target in ("all", "acestep"):
                        results["acestep"] = await asyncio.to_thread(pool.warm_up_acestep)
                    _write(writer, req_id, result={"status": "warm", "workers": results})
                except Exception as e:
                    _write(writer, req_id, error={"code": "WARMUP_FAILED", "message": str(e)})
            elif method == "render":
                # Beat filter: skip non-visual beats before expensive GPU work
                if params.get("narration") and params.get("game_state"):
                    from sidequest_daemon.renderer.beat_filter import should_generate
                    from sidequest_daemon.types import GameState, CombatState, ChaseState, Character

                    gs_raw = params["game_state"]
                    game_state = GameState(
                        location=gs_raw.get("location", ""),
                        time_of_day=gs_raw.get("time_of_day", ""),
                        characters=[Character(name=c.get("name", "")) for c in gs_raw.get("characters", [])],
                        combat=CombatState(in_combat=gs_raw.get("combat", {}).get("in_combat", False)),
                        chase=ChaseState(in_chase=gs_raw.get("chase", {}).get("in_chase", False)),
                    )
                    previous_location = params.get("previous_location")
                    if not should_generate(params["narration"], game_state, previous_location):
                        log.info("beat_filter: skipping non-visual beat")
                        _write(writer, req_id, result={"status": "skipped", "reason": "beat_filter"})
                        continue

                # If narration is provided, use SceneInterpreter for fast rule-based
                # StageCue extraction, then fall back to LLM subject extraction.
                if params.get("narration") and not params.get("positive_prompt"):
                    from sidequest_daemon.scene_interpreter import SceneInterpreter
                    from sidequest_daemon.types import GameState, Character

                    narrator_text = params["narration"]

                    # Extract documents and strip markers before visual processing
                    scene_interp = SceneInterpreter()
                    genre = params.get("genre", "unknown")
                    doc_events = scene_interp.extract_documents(narrator_text, genre=genre)
                    if doc_events:
                        log.info("scene_interpreter: extracted %d document(s)", len(doc_events))
                        params.setdefault("document_events", [])
                        for doc in doc_events:
                            params["document_events"].append(doc.model_dump())
                    narrator_text = scene_interp.strip_document_markers(narrator_text)
                    params["narration"] = narrator_text

                    # Try rule-based StageCue extraction (fast, no LLM)
                    gs_raw = params.get("game_state", {})
                    interp_state = GameState(
                        location=gs_raw.get("location", ""),
                        time_of_day=gs_raw.get("time_of_day", ""),
                        characters=[Character(name=c.get("name", "")) for c in gs_raw.get("characters", [])],
                    )
                    cues = scene_interp.interpret(narrator_text, interp_state)
                    if cues:
                        # Use the first cue's structured data instead of raw narration
                        top_cue = cues[0]
                        params["subject"] = top_cue.subject
                        params["mood"] = top_cue.mood
                        params["tags"] = top_cue.tags
                        params["tier"] = top_cue.tier.value
                        log.info(
                            "scene_interpreter — tier=%s subject=%s",
                            top_cue.tier.value,
                            top_cue.subject[:80],
                        )

                    # Fall back to LLM subject extraction if SceneInterpreter
                    # didn't produce a subject (or for refinement)
                    if not params.get("subject"):
                        from sidequest_daemon.media.subject_extractor import SubjectExtractor
                        extractor = SubjectExtractor()
                        extracted = await extractor.extract(params["narration"])
                        if not extracted or not extracted.get("subject"):
                            _write(writer, req_id, error={
                                "code": "EXTRACTION_FAILED",
                                "message": "SubjectExtractor returned no visual subject from narration. No fallback — refusing to render narrative prose directly.",
                            })
                            continue
                        # Build StageCue-compatible params from extraction
                        params["subject"] = extracted["subject"]
                        params["mood"] = extracted.get("mood", "")
                        params["tags"] = extracted.get("tags", [])
                        # Override tier if extractor found a better one
                        extracted_tier = extracted.get("tier", "")
                        if extracted_tier:
                            tier_lower = extracted_tier.lower()
                            if tier_lower in FLUX_TIERS:
                                params["tier"] = tier_lower
                        log.info(
                            "narration_extracted — subject=%s, mood=%s, tier=%s",
                            extracted["subject"][:80],
                            extracted.get("mood"),
                            params.get("tier"),
                        )

                # If we have visual_style + subject, compose through PromptComposer
                if params.get("subject") and params.get("art_style"):
                    from sidequest_daemon.media.prompt_composer import PromptComposer
                    from sidequest_daemon.renderer.models import RenderTier, StageCue
                    from sidequest_daemon.genre.models import VisualStyle

                    # Build a StageCue from params
                    tier_str = params.get("tier", "scene_illustration")
                    tier = RenderTier(tier_str) if tier_str in {t.value for t in RenderTier} else RenderTier.SCENE_ILLUSTRATION
                    cue = StageCue(
                        subject=params.get("subject", ""),
                        tier=tier,
                        location=params.get("location", ""),
                        mood=params.get("mood", ""),
                        characters=params.get("characters", []),
                        tags=params.get("tags", []),
                    )

                    # Build minimal VisualStyle from params
                    style = VisualStyle(
                        positive_suffix=params.get("art_style", ""),
                        negative_prompt=params.get("negative_prompt", ""),
                        preferred_model="flux",
                        visual_tag_overrides=params.get("visual_tag_overrides", {}),
                    )

                    composer = PromptComposer(
                        visual_tag_overrides=style.visual_tag_overrides,
                    )
                    composed = composer.compose(cue, style)
                    params["positive_prompt"] = composed.positive_prompt
                    params["clip_prompt"] = composed.clip_prompt
                    params["negative_prompt"] = composed.negative_prompt
                    params["seed"] = composed.seed
                    log.info(
                        "prompt_composed — positive=%s",
                        composed.positive_prompt[:150],
                    )

                # Serialize renders — only one GPU operation at a time
                async with render_lock:
                    try:
                        result = await asyncio.to_thread(pool.render, params)
                        _write(writer, req_id, result=result)
                    except Exception as e:
                        _write(writer, req_id, error={"code": "GENERATION_FAILED", "message": str(e)})
            elif method == "embed":
                # Story 15-7: Generate sentence embeddings for lore fragments
                text = params.get("text", "")
                if not text or not text.strip():
                    _write(writer, req_id, error={"code": "INVALID_REQUEST", "message": "embed requires non-empty 'text' field"})
                    continue
                try:
                    import time
                    start = time.monotonic()
                    worker = EmbedWorker()
                    embedding = worker.generate_embedding(text)
                    latency_ms = int((time.monotonic() - start) * 1000)
                    _write(writer, req_id, result={
                        "embedding": embedding,
                        "model": worker._model_name,
                        "latency_ms": latency_ms,
                    })
                except Exception as e:
                    _write(writer, req_id, error={"code": "EMBED_FAILED", "message": str(e)})
            else:
                _write(writer, req_id, error={"code": "UNKNOWN_METHOD", "message": f"Unknown: {method}"})
    except (ConnectionResetError, BrokenPipeError):
        log.info("Client disconnected: %s", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except BrokenPipeError:
            log.debug("Client already disconnected before wait_closed: %s", peer)
        except Exception:
            log.exception("Failed to close client writer")


def _write(
    writer: asyncio.StreamWriter,
    req_id: str,
    *,
    result: dict | None = None,
    error: dict | None = None,
) -> None:
    """Write a JSON response line to the client."""
    resp: dict = {"id": req_id}
    if result is not None:
        resp["result"] = result
    if error is not None:
        resp["error"] = error
    writer.write((json.dumps(resp) + "\n").encode())


async def _run_daemon(
    *,
    warmup: str | bool = False,
    output_dir: Path | None = None,
    genre_packs: Path | None = None,
) -> None:
    """Start the daemon server.

    warmup can be: False, True/"all", "flux", "tts"
    """
    if output_dir is None:
        env_dir = os.environ.get("SIDEQUEST_OUTPUT_DIR")
        output_dir = Path(env_dir) if env_dir else Path(tempfile.mkdtemp(prefix="sq-daemon-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if genre_packs is not None:
        os.environ["SIDEQUEST_GENRE_PACKS"] = str(genre_packs)
    pool = WorkerPool(output_dir)
    render_lock = asyncio.Lock()

    # Initialize media pipelines (audio + voice) via factory
    from sidequest_daemon.media.pipeline_factory import MediaPipelineFactory
    pipeline_factory = MediaPipelineFactory(
        audio_base_path=genre_packs,
    )
    # Audio init is deferred until a genre pack is loaded at session start.
    # The factory is stored on the pool so session handlers can call init_audio.
    pool.pipeline_factory = pipeline_factory
    log.info("MediaPipelineFactory initialized (audio pipeline deferred until session)")

    if warmup:
        target = warmup if isinstance(warmup, str) else "all"
        if target in ("all", "flux"):
            log.info("Pre-loading Flux model...")
            await asyncio.to_thread(pool.warm_up_flux)
        # ACE-Step removed — pre-recorded tracks used instead of procedural generation
        log.info("Models warm and ready")

    # Clean up stale socket
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    server = await asyncio.start_unix_server(
        lambda r, w: _handle_client(r, w, pool, render_lock),
        path=str(SOCKET_PATH),
    )

    # Write PID file
    PID_PATH.write_text(str(os.getpid()))
    log.info("Daemon listening on %s (pid %d)", SOCKET_PATH, os.getpid())
    log.info("Workers: %s", pool.status())

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _shutdown_signal() -> None:
        log.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown_signal)

    try:
        await stop_event.wait()
    finally:
        log.info("Shutting down daemon...")
        server.close()
        await server.wait_closed()
        pool.cleanup()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_PATH.exists():
            PID_PATH.unlink()
        log.info("Daemon stopped")


async def send_shutdown() -> None:
    """Send shutdown command to a running daemon."""
    if not SOCKET_PATH.exists():
        print("No daemon running (socket not found)")
        sys.exit(1)

    try:
        reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
        req = json.dumps({"id": "shutdown", "method": "shutdown", "params": {}})
        writer.write((req + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        resp = json.loads(line.decode())
        if resp.get("result", {}).get("status") == "ok":
            print("Daemon shutdown requested")
        else:
            print(f"Unexpected response: {resp}")
        writer.close()
    except (ConnectionRefusedError, FileNotFoundError):
        print("Daemon not responding — cleaning up stale socket")
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_PATH.exists():
            PID_PATH.unlink()


async def send_status() -> None:
    """Query daemon status."""
    if not SOCKET_PATH.exists():
        print("No daemon running (socket not found)")
        sys.exit(1)

    try:
        reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
        req = json.dumps({"id": "status", "method": "status", "params": {}})
        writer.write((req + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        resp = json.loads(line.decode())
        if "result" in resp:
            status = resp["result"]
            print(f"Flux: {status.get('flux', 'unknown')}")
            print(f"ACE-Step: {status.get('acestep', 'unknown')}")
            tiers = status.get("supported_tiers", {})
            print(f"Flux tiers: {', '.join(tiers.get('flux', []))}")
        else:
            print(f"Error: {resp.get('error', resp)}")
        writer.close()
    except (ConnectionRefusedError, FileNotFoundError):
        print("Daemon not responding")


def _parse_arg(name: str) -> str | None:
    """Extract --name VALUE from sys.argv, return value or None."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def main() -> None:
    """CLI entry point for sidequest-renderer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if "--shutdown" in sys.argv:
        asyncio.run(send_shutdown())
    elif "--status" in sys.argv:
        asyncio.run(send_status())
    else:
        # Warmup is the default — use --no-warmup to skip (e.g. for testing)
        warmup: str | bool = "all"
        for arg in sys.argv[1:]:
            if arg == "--no-warmup":
                warmup = False
            elif arg.startswith("--warmup="):
                warmup = arg.split("=", 1)[1]

        # Parse optional paths
        genre_packs_str = _parse_arg("--genre-packs")
        output_dir_str = _parse_arg("--output-dir")
        genre_packs = Path(genre_packs_str) if genre_packs_str else None
        output_dir = Path(output_dir_str) if output_dir_str else None

        asyncio.run(_run_daemon(
            warmup=warmup,
            output_dir=output_dir,
            genre_packs=genre_packs,
        ))


if __name__ == "__main__":
    main()
