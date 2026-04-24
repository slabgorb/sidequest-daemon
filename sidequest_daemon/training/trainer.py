"""Thin wrapper over mlx-lm fine-tune, used by sidequest-train CLI (ADR-073 Phase 3).

The mlx-lm invocation is dependency-injected via `trainer_fn` so tests can
exercise the surrounding plumbing without spinning GPU work.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sidequest_daemon.training.corpus_loader import TrainingPair
from sidequest_daemon.training.format import format_for_qwen

TrainerFn = Callable[..., dict]


@dataclass
class TrainerConfig:
    pairs: list[TrainingPair]
    base_model: str
    out_dir: Path
    iters: int
    batch_size: int
    genre: str | None = None
    include_genre_tag: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class TrainingResult:
    adapter_path: Path
    stats: dict


def _default_trainer(**kwargs) -> dict:
    # Lazy import — mlx-lm is an optional extra, not imported at daemon boot.
    from mlx_lm.lora import run as mlx_lora_run  # type: ignore[import-not-found]

    return mlx_lora_run(**kwargs)


def run_training(cfg: TrainerConfig, *, trainer_fn: TrainerFn | None = None) -> TrainingResult:
    if not cfg.pairs:
        raise ValueError("run_training called with empty pairs list")

    ts = time.strftime("%Y%m%d-%H%M%S")
    model_slug = cfg.base_model.split("/")[-1]
    adapter_path = cfg.out_dir / f"{model_slug}-{ts}"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Write the ChatML JSONL that mlx-lm reads.
    train_jsonl = adapter_path / "train.jsonl"
    with train_jsonl.open("w") as fh:
        for pair in cfg.pairs:
            fh.write(json.dumps(format_for_qwen(pair, include_genre_tag=cfg.include_genre_tag)))
            fh.write("\n")

    fn = trainer_fn or _default_trainer
    stats = fn(
        model=cfg.base_model,
        data=str(train_jsonl.parent),
        adapter_path=str(adapter_path),
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        **cfg.extra,
    )
    return TrainingResult(adapter_path=adapter_path, stats=stats)
