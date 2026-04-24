# Training Pipeline (ADR-073 Phase 3)

`sidequest-train` fine-tunes a QLoRA adapter against Group D corpus data.

## Install

```
cd sidequest-daemon
uv sync --extra training
```

This adds `mlx-lm>=0.20` (macOS/Apple Silicon only).

## Usage

```
uv run sidequest-train \
  --corpus ~/.sidequest/corpus/mined/*.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct \
  --out ~/.sidequest/loras/caverns_and_claudes \
  --genre caverns_and_claudes \
  --include-genre-tag \
  --iters 600 \
  --batch-size 4
```

Flags:
- `--corpus` — one or more JSONL files from `~/.sidequest/corpus/mined/`
- `--base` — HF model id passed straight to `mlx-lm`
- `--out` — adapters land under `<out>/<model-slug>-<ts>/adapters.safetensors`
- `--genre` — filter corpus to a single genre slug (optional)
- `--include-genre-tag` — prepend `[genre=X]` to system prompt in training data
- `--iters`, `--batch-size`, `--seed` — standard mlx-lm knobs
- `--smoke` — skip real training; write a dummy adapter (used in CI and for pipeline shake-down)

## Corpus volume

ADR-073 calls for ~5K pairs minimum for base fine-tune, ~500 per genre for
LoRA. Below 500 total the CLI prints a loud overfit warning — training still
runs, but the adapter will not generalise.

## Deployment (not wired in Group E)

The output `adapters.safetensors` is an MLX adapter. Serving it via Ollama
requires GGUF conversion, which is explicitly out of scope for Group E.
Deployment will be addressed in a future plan.
