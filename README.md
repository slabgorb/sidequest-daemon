# sidequest-daemon

Media services daemon for SideQuest — image generation, text-to-speech, and audio playback. Runs as a standalone Python process communicating via Unix socket, providing rendering services to the Rust game engine (`sidequest-api`).

## Architecture

```
sidequest-api (Rust)  ──JSON over Unix socket──►  sidequest-daemon (Python)
                                                    ├── Flux image generation (MPS/CUDA)
                                                    ├── Kokoro / Piper TTS
                                                    ├── Audio mixer (pygame)
                                                    └── Scene interpretation
```

The daemon is a hot-loaded process — start it once, and it keeps GPU models warm between requests. The Rust backend sends render commands over `/tmp/sidequest-renderer.sock` and gets back file paths to generated images/audio.

## Prerequisites

- Python 3.11+
- For image generation: Apple Silicon (MPS) or NVIDIA GPU (CUDA)
- For TTS: Kokoro ONNX models auto-download to `~/.sidequest/models/kokoro/`
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

# Warm up only Flux (skip TTS):
sidequest-renderer --warmup=flux
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
| `render` | Generate an image from a StageCue |
| `warm_up` | Pre-load GPU models (`{"worker": "flux"\|"tts"\|"all"}`) |
| `shutdown` | Graceful daemon shutdown |

## Package structure

```
sidequest_daemon/
├── media/           # Daemon server, workers, cache, prompt composer
│   ├── daemon.py    # Entry point — Unix socket server + CLI
│   ├── workers/     # Flux, TTS, ACE-Step worker processes
│   └── ...
├── renderer/        # Data models (StageCue, RenderTier, RenderResult)
├── audio/           # Mixer, library backend, theme rotation
├── voice/           # Kokoro TTS, Piper fallback, voice routing
├── genre/           # Genre pack model subset (VisualStyle, AudioConfig)
├── ml/              # GPU memory management
├── scene_interpreter.py  # Narrative → StageCue rules engine
├── config.py        # Path resolution helpers
└── types.py         # Stub types for game-engine interfaces
```

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
