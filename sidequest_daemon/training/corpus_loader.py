"""Read Group D `TrainingPair` JSONL into typed rows (ADR-073 Phase 3)."""
from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class MineProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_save: str
    event_seq: int | None


class TrainingPair(BaseModel):
    """Locally-mirrored TrainingPair schema from sidequest-server/sidequest/corpus.

    We keep a copy instead of importing to avoid a daemon → server dep.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    genre: str
    world: str
    round_number: int = Field(ge=0)
    input_text: str = Field(min_length=1)
    output_text: str = Field(min_length=1)
    provenance: MineProvenance


@dataclass(frozen=True)
class CorpusStats:
    total: int
    by_genre: dict[str, int]
    min_round: int
    max_round: int


def load_training_pairs(paths: Iterable[Path]) -> Iterator[TrainingPair]:
    """Yield TrainingPair rows from JSONL files; skip-and-log malformed lines."""
    for path in paths:
        with Path(path).open() as fh:
            for idx, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("corpus.bad_json path=%s line=%d err=%s", path, idx, exc)
                    continue
                try:
                    yield TrainingPair.model_validate(obj)
                except ValidationError as exc:
                    logger.warning("corpus.bad_schema path=%s line=%d err=%s", path, idx, exc)


def summarise(pairs: Iterable[TrainingPair]) -> CorpusStats:
    pairs = list(pairs)
    if not pairs:
        return CorpusStats(total=0, by_genre={}, min_round=0, max_round=0)
    by_genre = Counter(p.genre for p in pairs)
    rounds = [p.round_number for p in pairs]
    return CorpusStats(
        total=len(pairs),
        by_genre=dict(by_genre),
        min_round=min(rounds),
        max_round=max(rounds),
    )
