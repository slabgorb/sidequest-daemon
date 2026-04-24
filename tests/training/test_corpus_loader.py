"""Tests for corpus_loader — reading Group D mined JSONL."""
from __future__ import annotations

import json
from pathlib import Path

from sidequest_daemon.training.corpus_loader import (
    CorpusStats,
    load_training_pairs,
    summarise,
)


FIXTURE = Path(__file__).parent / "fixtures" / "mined_sample.jsonl"


def test_load_training_pairs_yields_all_rows():
    pairs = list(load_training_pairs([FIXTURE]))
    assert len(pairs) == 3
    assert {p.genre for p in pairs} == {"caverns_and_claudes", "heavy_metal"}


def test_load_training_pairs_skips_malformed(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n" + FIXTURE.read_text())
    pairs = list(load_training_pairs([bad]))
    # First line skipped, remaining 3 loaded.
    assert len(pairs) == 3


def test_load_training_pairs_skips_empty_text(tmp_path):
    bad = tmp_path / "empty.jsonl"
    row = json.loads(FIXTURE.read_text().splitlines()[0])
    row["input_text"] = ""
    bad.write_text(json.dumps(row) + "\n")
    pairs = list(load_training_pairs([bad]))
    assert pairs == []


def test_summarise_counts_by_genre():
    stats = summarise(load_training_pairs([FIXTURE]))
    assert isinstance(stats, CorpusStats)
    assert stats.total == 3
    assert stats.by_genre == {"caverns_and_claudes": 2, "heavy_metal": 1}
    assert stats.min_round <= stats.max_round
