"""`sidequest-train` CLI — train a genre LoRA adapter (ADR-073 Phase 3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sidequest_daemon.training.corpus_loader import (
    CorpusStats,
    load_training_pairs,
    summarise,
)
from sidequest_daemon.training.trainer import (
    TrainerConfig,
    TrainingResult,
    run_training,
)

LOW_VOLUME_THRESHOLD = 500


def _fake_smoke_trainer(**kwargs: Any) -> dict:
    """Smoke-mode stand-in: writes a dummy adapter + returns fake stats."""
    adapter_path = Path(kwargs["adapter_path"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapters.safetensors").write_bytes(b"\x00" * 32)
    return {
        "final_train_loss": None,
        "final_val_loss": None,
        "iters": kwargs.get("iters", 0),
        "mode": "smoke",
    }


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sidequest-train")
    p.add_argument("--corpus", nargs="+", required=True, type=Path,
                   help="Group D mined JSONL file(s).")
    p.add_argument("--base", required=True,
                   help="Base HF model id (e.g. Qwen/Qwen2.5-7B-Instruct).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for adapters.")
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--genre", default=None, help="Filter corpus to this genre slug.")
    p.add_argument("--include-genre-tag", action="store_true",
                   help="Prepend [genre=X] to system prompt in training data.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="Use a fake trainer that writes a dummy adapter. No GPU work.")
    return p.parse_args(argv)


def _print_stats(stats: CorpusStats) -> None:
    print(
        f"corpus: total={stats.total} min_round={stats.min_round} "
        f"max_round={stats.max_round}"
    )
    for genre, n in sorted(stats.by_genre.items()):
        print(f"  {genre}: {n}")


def _warn_low_volume(stats: CorpusStats) -> None:
    if 0 < stats.total < LOW_VOLUME_THRESHOLD:
        print(
            f"WARNING: training on {stats.total} pairs; "
            f"ADR-073 recommends 5K+ for base fine-tune. "
            f"Output adapter is expected to overfit.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv if argv is not None else sys.argv[1:])
    pairs = list(load_training_pairs(args.corpus))
    if args.genre is not None:
        pairs = [p for p in pairs if p.genre == args.genre]
    stats = summarise(pairs)
    _print_stats(stats)
    _warn_low_volume(stats)

    if stats.total == 0:
        print(
            "ERROR: no training pairs after loading/filtering; aborting.",
            file=sys.stderr,
        )
        return 2

    cfg = TrainerConfig(
        pairs=pairs,
        base_model=args.base,
        out_dir=args.out,
        iters=args.iters,
        batch_size=args.batch_size,
        genre=args.genre,
        include_genre_tag=args.include_genre_tag,
        seed=args.seed,
    )
    trainer_fn = _fake_smoke_trainer if args.smoke else None
    result: TrainingResult = run_training(cfg, trainer_fn=trainer_fn)
    print(f"adapter written to: {result.adapter_path}")
    print(f"stats: {result.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
