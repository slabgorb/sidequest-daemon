# sidequest-daemon

Media services daemon for SideQuest — image generation and music generation.
Runs as a standalone Python process communicating via Unix socket, providing
rendering services to the Python `sidequest-server` (port 8765, ADR-082).

> **Active renderer:** Z-Image Turbo (MLX). Flux paths have been retired —
> `zimage_mlx_worker.py` is the sole runtime image worker.
>
> **Music tier (ADR-095):** ACE-Step generation runs on operator command, not
> at every turn. Per-track `*_input_params.json` files in
> `sidequest-content/genre_packs/<pack>/audio/music/` are the canonical spec;
> the daemon produces OGG (libopus 96k) and uploads to R2.

## Architecture

```
sidequest-server (Python)  ──JSON over Unix socket──►  sidequest-daemon (Python)
                                                         ├── Z-Image image generation (MLX, Apple Silicon)
                                                         ├── ACE-Step music generation (operator-triggered)
                                                         ├── Audio playback (pygame-ce mixer)
                                                         └── Scene interpretation (narration → StageCue)
```

The daemon is a hot-loaded process — start it once, and it keeps the Z-Image
model warm between requests. The server sends render commands over
`/tmp/sidequest-renderer.sock` and gets back file paths to generated images.

## Services

**Z-Image Worker** (`media/workers/zimage_mlx_worker.py`, ADR-070) — Image
generation across composition tiers from ADR-086: portrait, POI landscape,
illustration. Targets Apple Silicon via MLX / mflux. Z-Image Turbo is the
active model; prose bleeds as text in image prompts, so no negative-prompt
syntax is used. See `sidequest-content/PROMPTING_Z_IMAGE.md`.

**Music Pipeline** (`media/music_pipeline.py`, ADR-095) — ACE-Step generation
tier. Operator runs `python scripts/generate_music.py --genre <pack>` from the
orchestrator; the script discovers `*_input_params.json` files and dispatches
each to the daemon as `tier=music`. The daemon runs ACE-Step → ffmpeg
WAV→OGG (libopus 96k) → R2 upload at
`genre_packs/<pack>/audio/music/<track>.ogg`. Generation is **operator-
triggered**, not per-turn — costly and slow, so it's a build-time act.

**Audio Playback** (`audio/`) — `pygame-ce` mixer with music + SFX channels.
TTS / voice / ducking paths are gone (2026-04).

**Scene Interpreter** (`scene_interpreter.py`) — Rules-based narration-to-
`StageCue` extractor. Turns narrator prose into structured visual cues for the
image pipeline.

**Subject Extractor** (`media/subject_extractor.py`) — Claude CLI invocation
that turns prose into visual descriptions for portrait/POI prompts.

## Prerequisites

- Python 3.12 (mflux pins below 3.13)
- Apple Silicon (MLX / MPS). CUDA path is no longer the primary target after ADR-070.
- Genre packs directory (shared YAML + audio assets)

## Installation

```bash
# From sidequest-daemon:
uv sync

# Or from the orchestrator root:
just setup
```

## Usage

### Starting the daemon

```bash
# Lazy model loading — no GPU needed until first render:
uv run python -m sidequest_daemon

# With model warmup (recommended for gameplay):
uv run python -m sidequest_daemon --warmup
```

### From the orchestrator

```bash
just daemon          # Start with warmup
just daemon-status   # Check if running
just daemon-stop     # Graceful shutdown
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SIDEQUEST_GENRE_PACKS` | `../sidequest-content/genre_packs` | Path to genre packs |
| `SIDEQUEST_OUTPUT_DIR` | temp dir | Directory for generated output files |

## Protocol

Newline-delimited JSON over Unix domain socket at `/tmp/sidequest-renderer.sock`.

### Request

```json
{"id": "unique-id", "method": "render", "params": {"tier": "portrait", "subject": "...", ...}}
```

### Response

```json
{"id": "unique-id", "result": {"image_path": "/tmp/...", "width": 1024, "height": 1024, "elapsed_ms": 2800}}
```

### Methods

| Method | Description |
|--------|-------------|
| `ping` | Health check — returns `{"status": "ok"}` |
| `status` | Worker pool status and loaded models |
| `render` | Generate an image, routed by `tier` |
| `music` | Generate a music track via ACE-Step (ADR-095) — operator-triggered |
| `warm_up` | Pre-load the Z-Image model (and embedding model for ADR-048 lore RAG) |
| `shutdown` | Graceful daemon shutdown |

## Package structure

```
sidequest_daemon/
├── media/                       # Daemon server, workers, pipelines
│   ├── daemon.py                # Entry point — Unix socket server + CLI
│   ├── workers/
│   │   └── zimage_mlx_worker.py # Sole runtime image worker (ADR-070)
│   ├── music_pipeline.py        # ACE-Step → ffmpeg → R2 (ADR-095)
│   ├── ace_step_adapter.py
│   ├── prompt_composer.py       # Tier-aware prompt prefixes, token budgeting
│   ├── subject_extractor.py     # Prose → visual description via Claude CLI
│   ├── recipes.py / recipe_loader.py
│   ├── catalogs.py / camera_specs.py
│   ├── post_processor.py        # Post-render adjustments
│   ├── preview.py
│   ├── r2_writer.py             # R2 upload for music artifacts
│   ├── gpu_detect.py
│   └── zimage_config.py
├── renderer/                    # Data models (StageCue, RenderTier, RenderResult)
├── audio/                       # Mixer (pygame-ce), library backend, scene rotation
├── genre/                       # Genre pack model subset (VisualStyle, AudioConfig)
├── ml/                          # GPU memory management (ADR-046)
└── scene_interpreter.py         # Narrative → StageCue rules engine
```

> If a file listed above has moved or been removed, treat the source tree as
> authoritative and update this README rather than the other way around.

## Development

```bash
just daemon-test     # Run tests
just daemon-lint     # ruff check
```

Smoke test without GPU:

```bash
uv run pytest tests/test_daemon_smoke.py -v
```

## Origin

Lifted from [sq-2](https://github.com/slabgorb/sq-2) (the original Python
SideQuest codebase) as part of the now-archived Rust rewrite. The daemon
boundary survived the Rust→Python port (ADR-082) because the isolation
benefits — independent restart, GPU lifecycle, slow model warmup — apply
regardless of server language.

## Training pipeline

See `docs/training.md` for `sidequest-train` usage (ADR-073 Phase 3).

## Related repos

- [orc-quest](https://github.com/slabgorb/orc-quest) — Orchestrator, ADRs
- [sidequest-server](https://github.com/slabgorb/sidequest-server) — Python FastAPI backend
- [sidequest-ui](https://github.com/slabgorb/sidequest-ui) — React client
- [sidequest-content](https://github.com/slabgorb/sidequest-content) — Genre packs
