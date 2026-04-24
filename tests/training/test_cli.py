"""Tests for sidequest-train CLI — exercised in --smoke mode only."""
from __future__ import annotations

import json
from pathlib import Path

from sidequest_daemon.training.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "mined_sample.jsonl"


def test_cli_smoke_uses_fake_trainer(tmp_path, capsys):
    out = tmp_path / "loras"
    rc = main([
        "--corpus", str(FIXTURE),
        "--base", "Qwen/Qwen2.5-7B-Instruct",
        "--out", str(out),
        "--iters", "1",
        "--batch-size", "1",
        "--smoke",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "total=3" in captured
    assert "caverns_and_claudes" in captured
    # Smoke mode writes a dummy adapter.
    adapters = list(out.rglob("adapters.safetensors"))
    assert len(adapters) == 1


def test_cli_low_volume_warning(tmp_path, capsys):
    tiny = tmp_path / "tiny.jsonl"
    row = json.loads(FIXTURE.read_text().splitlines()[0])
    tiny.write_text(json.dumps(row) + "\n")
    rc = main([
        "--corpus", str(tiny),
        "--base", "Qwen/Qwen2.5-7B-Instruct",
        "--out", str(tmp_path / "loras"),
        "--iters", "1",
        "--batch-size", "1",
        "--smoke",
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "ADR-073 recommends 5K+" in err


def test_cli_empty_corpus_nonzero_exit(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    rc = main([
        "--corpus", str(empty),
        "--base", "Qwen/Qwen2.5-7B-Instruct",
        "--out", str(tmp_path / "loras"),
        "--iters", "1",
        "--batch-size", "1",
        "--smoke",
    ])
    assert rc == 2
