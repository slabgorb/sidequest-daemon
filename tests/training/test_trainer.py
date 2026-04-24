"""Tests for trainer.py — uses a FakeTrainer, does NOT run real MLX."""
from __future__ import annotations

from pathlib import Path

import pytest

from sidequest_daemon.training.corpus_loader import MineProvenance, TrainingPair
from sidequest_daemon.training.trainer import TrainerConfig, run_training


def _pair(i: int = 0) -> TrainingPair:
    return TrainingPair(
        schema_version=1,
        genre="g",
        world="w",
        round_number=i,
        input_text=f"in {i}",
        output_text=f"out {i}",
        provenance=MineProvenance(source_save="/tmp/s.db", event_seq=i),
    )


def test_run_training_invokes_trainer_and_writes_adapter(tmp_path):
    calls: list[dict] = []

    def fake_trainer(**kwargs) -> dict:
        calls.append(kwargs)
        adapter_path = Path(kwargs["adapter_path"])
        adapter_path.mkdir(parents=True, exist_ok=True)
        (adapter_path / "adapters.safetensors").write_bytes(b"\x00" * 8)
        return {"final_train_loss": 1.23, "final_val_loss": 1.45, "iters": kwargs["iters"]}

    cfg = TrainerConfig(
        pairs=[_pair(i) for i in range(4)],
        base_model="Qwen/Qwen2.5-7B-Instruct",
        out_dir=tmp_path / "loras" / "g",
        iters=3,
        batch_size=2,
    )
    result = run_training(cfg, trainer_fn=fake_trainer)

    assert (result.adapter_path / "adapters.safetensors").is_file()
    assert result.stats["iters"] == 3
    assert len(calls) == 1


def test_run_training_empty_pairs_raises():
    cfg = TrainerConfig(pairs=[], base_model="x", out_dir=Path("/tmp/unused"), iters=1, batch_size=1)
    with pytest.raises(ValueError):
        run_training(cfg, trainer_fn=lambda **_: {"iters": 0})
