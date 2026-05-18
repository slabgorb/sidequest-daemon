"""Deploy-pipeline CLI for story 48-3 — the non-test consumer that wires
corpus gate -> (train) -> GGUF convert -> Ollama Modelfile into one flow.

Mirrors the 48-4 ``ab_eval_harness_cli`` / 48-2 ``ollama_latency_check``
operator-evidence pattern: a two-layer split where the gate + argv plumbing
are CI-tested and the live conversion / ``ollama create`` are
operator-evidence only (Keith's M3 Ultra). When the conversion tooling is
absent the CLI is a graceful no-op with a documented operator note, never a
traceback.

Exit codes:
    0  success
    3  configuration error (missing corpus, below the hard corpus gate)
    4  conversion tooling unavailable — operator-evidence no-op; the live
       deploy must be run on the M3 Ultra (report records the note)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from sidequest_daemon.training.corpus_gate import (
    CorpusGateError,
    enforce_corpus_gate,
)
from sidequest_daemon.training.corpus_loader import load_training_pairs
from sidequest_daemon.training.gguf_convert import (
    GgufConversionError,
    convert_lora_to_gguf,
)
from sidequest_daemon.training.ollama_modelfile import (
    OllamaModelfileError,
    create_ollama_model,
)
from sidequest_daemon.training.trainer import (
    TrainerConfig,
    run_training,
)

logger = logging.getLogger(__name__)

EXIT_PASS = 0
EXIT_CONFIG_ERROR = 3
EXIT_TOOLING_UNAVAILABLE = 4

OPERATOR_NOTE = """## Deploy Pipeline — Operator Evidence Only

**Status:** Conversion tooling unavailable on this host.

The MLX -> GGUF conversion and `ollama create` steps are operator-evidence
only. They must be run on Keith's M3 Ultra, the only host with llama.cpp +
a live Ollama serving the local model.

To collect operator evidence:
1. On the M3 Ultra, run this CLI with llama.cpp + Ollama present.
2. Save the markdown report to the PR.
3. GM-panel A/B review (story 48-4 harness) is the quality criterion.
"""


def _smoke_trainer(**kwargs: Any) -> dict:
    """Fake trainer (no GPU): writes a dummy adapter, mirrors cli.py."""
    adapter_path = Path(kwargs["adapter_path"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapters.safetensors").write_bytes(b"\x00" * 32)
    return {"mode": "smoke", "iters": kwargs.get("iters", 0)}


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sidequest-deploy")
    p.add_argument("--corpus", nargs="+", required=True, type=Path)
    p.add_argument("--base", required=True)
    p.add_argument("--genre", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse(argv if argv is not None else sys.argv[1:])
    except SystemExit:
        return EXIT_CONFIG_ERROR

    missing = [str(p) for p in args.corpus if not Path(p).is_file()]
    if missing:
        logger.error("deploy.corpus_missing paths=%s", missing)
        return EXIT_CONFIG_ERROR

    pairs = [
        pr
        for pr in load_training_pairs(args.corpus)
        if pr.genre == args.genre
    ]

    try:
        enforce_corpus_gate(len(pairs))
    except CorpusGateError as exc:
        logger.error("deploy.corpus_gate_failed %s", exc)
        return EXIT_CONFIG_ERROR

    cfg = TrainerConfig(
        pairs=pairs,
        base_model=args.base,
        out_dir=args.out,
        iters=args.iters,
        batch_size=args.batch_size,
        genre=args.genre,
    )
    trainer_fn = _smoke_trainer if args.smoke else None
    result = run_training(cfg, trainer_fn=trainer_fn)

    gguf_out = Path(args.out) / f"sidequest-narrator-{args.genre}.gguf"
    try:
        convert_lora_to_gguf(result.adapter_path, gguf_out)
        tag = create_ollama_model(
            args.genre,
            args.base,
            gguf_out,
            modelfile_dir=Path(args.out) / "modelfile",
        )
    except (
        FileNotFoundError,
        GgufConversionError,
        OllamaModelfileError,
    ) as exc:
        logger.warning("deploy.tooling_unavailable %s", exc)
        _write(args.output_md, OPERATOR_NOTE)
        return EXIT_TOOLING_UNAVAILABLE

    _write(
        args.output_md,
        f"# Deploy Report\n\nGenre: {args.genre}\n"
        f"Model: {tag}\nAdapter: {result.adapter_path}\nGGUF: {gguf_out}\n",
    )
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
