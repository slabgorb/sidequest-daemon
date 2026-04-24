"""Tests for training format — ChatML for Qwen 2.5."""

from __future__ import annotations

from sidequest_daemon.training.corpus_loader import MineProvenance, TrainingPair
from sidequest_daemon.training.format import (
    QWEN_SYSTEM_PROMPT,
    format_for_qwen,
)


def _pair(input_text="in", output_text="out") -> TrainingPair:
    return TrainingPair(
        schema_version=1,
        genre="caverns_and_claudes",
        world="forge",
        round_number=0,
        input_text=input_text,
        output_text=output_text,
        provenance=MineProvenance(source_save="/tmp/s.db", event_seq=1),
    )


def test_format_for_qwen_emits_messages():
    out = format_for_qwen(_pair())
    assert out == {
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": "in"},
            {"role": "assistant", "content": "out"},
        ]
    }


def test_format_includes_genre_tag_when_asked():
    out = format_for_qwen(_pair(), include_genre_tag=True)
    # Genre is prepended to the system prompt.
    assert out["messages"][0]["content"].startswith("[genre=caverns_and_claudes]")
