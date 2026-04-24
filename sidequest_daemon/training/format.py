"""Emit ChatML training examples for Qwen 2.5 mlx-lm fine-tune (ADR-073 Phase 3)."""

from __future__ import annotations

from sidequest_daemon.training.corpus_loader import TrainingPair

QWEN_SYSTEM_PROMPT = (
    "You are the SideQuest narrator. Respond with in-world prose grounded in "
    "game state. Obey genre truth. Never narrate for the player."
)


def format_for_qwen(pair: TrainingPair, *, include_genre_tag: bool = False) -> dict:
    system = QWEN_SYSTEM_PROMPT
    if include_genre_tag:
        system = f"[genre={pair.genre}] {system}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": pair.input_text},
            {"role": "assistant", "content": pair.output_text},
        ]
    }
