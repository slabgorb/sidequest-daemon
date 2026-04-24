# sidequest-daemon

Media services daemon for SideQuest — image generation and audio playback.
Runs as a standalone Python process communicating via Unix socket, providing
rendering services to the Rust game engine (`sidequest-api`).

> **2026-04 note:** Kokoro TTS and runtime ACE-Step music generation have been
> removed from this daemon. Music is now pre-rendered at build time and played
> back from a library (see the `audio/` package). TTS, Piper, and voice
> synthesis paths no longer exist.

## Architecture

```
sidequest-api (Rust)  ──JSON over Unix socket──►  sidequest-daemon (Python)
                                                    ├── Flux image generation (MLX / MPS / CUDA)
                                                    ├── Audio library playback (pygame-ce)
                                                    └── Scene interpretation
```

The daemon is a hot-loaded process — start it once, and it keeps the Flux image
model warm between requests. The Rust backend sends render commands over
`/tmp/sidequest-renderer.sock` and gets back file paths to generated images.

## Services

**Flux Worker** — Image generation using Flux schnell and dev models. Six
render tiers: `scene_illustration`, `portrait`, `landscape`, `text_overlay`,
`cartography` (`tactical_sketch` retired; see ADR-086). Current backend is the MLX worker
(`workers/flux_mlx_worker.py`, per ADR-070), targeting Apple Silicon. Earlier
PyTorch/diffusers code is retired.

**Audio Library Backend** — Reads pre-rendered music tracks (ACE-Step produced
at build time, not at runtime) from genre pack audio directories and selects
tracks by mood via `rotator.py`. Playback uses `pygame-ce` through
`audio/mixer.py` across the channels defined in that module.

**Scene Interpreter** — Rules-based narration-to-`StageCue` extractor. Turns
narrator prose into structured visual cues that the image pipeline can render.

## Prerequisites

- Python 3.11+
- For image generation: Apple Silicon (MLX / MPS) preferred; NVIDIA CUDA path is
  no longer the primary target after ADR-070
- Genre packs directory (shared YAML + audio assets)

## Installation

```bash
# From the sidequest-daemon directory:
pip install -e ".[dev]"

# Or from the orchestrator root:
just daemon-install
```

## Usage

### Starting the daemon

```bash
# Basic start (lazy model loading — no GPU needed until first render):
sidequest-renderer

# Start with GPU models pre-loaded (recommended for gameplay):
sidequest-renderer --warmup

# Warm up a single worker (skip others):
sidequest-renderer --warmup=flux
sidequest-renderer --warmup=embed
```

### From the orchestrator

```bash
just daemon-run      # Start daemon
just daemon-warmup   # Start with model warmup
just daemon-status   # Check if running
just daemon-stop     # Graceful shutdown
```

### Daemon management

```bash
sidequest-renderer --status    # Check daemon status
sidequest-renderer --shutdown  # Graceful shutdown
```

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIDEQUEST_GENRE_PACKS` | `../genre_packs` (sibling dir) | Path to genre packs directory |
| `SIDEQUEST_OUTPUT_DIR` | temp dir | Directory for generated output files |

### CLI arguments

```bash
sidequest-renderer --genre-packs /path/to/genre_packs --output-dir /path/to/output
```

## Protocol

The daemon communicates via newline-delimited JSON over a Unix domain socket at `/tmp/sidequest-renderer.sock`.

### Request format

```json
{"id": "unique-id", "method": "render", "params": {"tier": "portrait", "subject": "a weathered orc warrior", ...}}
```

### Response format

```json
{"id": "unique-id", "result": {"image_path": "/tmp/sq-daemon-xxx/render.png", "width": 768, "height": 1024, "elapsed_ms": 4200}}
```

### Methods

| Method | Description |
|--------|-------------|
| `ping` | Health check — returns `{"status": "ok"}` |
| `status` | Worker pool status and loaded models |
| `render` | Generate an image (routed by tier) |
| `warm_up` | Pre-load the Flux model (and embedding model for ADR-048 lore RAG) |
| `shutdown` | Graceful daemon shutdown |

## Package structure

```
sidequest_daemon/
├── media/           # Daemon server, workers, cache, prompt composer
│   ├── daemon.py    # Entry point — Unix socket server + CLI
│   ├── workers/     # flux_mlx_worker.py (sole runtime worker)
│   └── ...
├── renderer/        # Data models (StageCue, RenderTier, RenderResult)
├── audio/           # Mixer, library backend, theme rotation, scene interpreter
├── genre/           # Genre pack model subset (VisualStyle, AudioConfig)
├── ml/              # GPU memory management
├── scene_interpreter.py  # Narrative → StageCue rules engine
└── types.py         # Stub types for game-engine interfaces
```

> If a file listed above has moved or been removed, treat the source tree as
> authoritative and update this README rather than the other way around.

## Development

```bash
# Run tests
just daemon-test

# Lint
just daemon-lint

# Smoke test (no GPU required)
pytest tests/test_daemon_smoke.py -v
```

## Origin

Lifted from [sq-2](https://github.com/slabgorb/sq-2) (the original Python SideQuest codebase) as part of the Rust rewrite. The daemon code was already cleanly separated behind a Unix socket boundary — this repo extracts it into a standalone package.

## Training pipeline

See `docs/training.md` for `sidequest-train` usage (ADR-073 Phase 3).
