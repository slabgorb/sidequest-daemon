"""Hard corpus precondition gate for story 48-3 substep (a).

Distinct from ``cli._warn_low_volume`` (a *soft* stderr warning that trains
anyway): this is a *hard* gate. Below the threshold the caller must refuse
to ship an overfit adapter and surface the decision loudly (project rule:
No Silent Fallbacks) rather than silently defaulting to a doomed run.
"""
from __future__ import annotations

from dataclasses import dataclass

CORPUS_GATE_MIN_PAIRS = 500


class CorpusGateError(RuntimeError):
    """Raised when the mined corpus is below the training gate threshold."""


@dataclass(frozen=True)
class CorpusGateResult:
    passed: bool
    total: int
    threshold: int
    reason: str


def evaluate_corpus_gate(
    total: int, *, min_pairs: int = CORPUS_GATE_MIN_PAIRS
) -> CorpusGateResult:
    """Return the gate verdict for ``total`` mined TrainingPair entries.

    Passing requires ``total >= min_pairs`` (the boundary value passes).
    """
    passed = total >= min_pairs
    if passed:
        reason = f"{total} pairs >= gate {min_pairs}"
    else:
        reason = (
            f"only {total} mined pairs; gate requires {min_pairs}. "
            f"Gate on more playtests rather than ship an overfit adapter."
        )
    return CorpusGateResult(
        passed=passed, total=total, threshold=min_pairs, reason=reason
    )


def enforce_corpus_gate(
    total: int, *, min_pairs: int = CORPUS_GATE_MIN_PAIRS
) -> CorpusGateResult:
    """Like :func:`evaluate_corpus_gate` but raises loudly when not passed.

    This is the No-Silent-Fallbacks entry point: an insufficient corpus is
    a hard stop, never a silent default into an overfit training run.
    """
    result = evaluate_corpus_gate(total, min_pairs=min_pairs)
    if not result.passed:
        raise CorpusGateError(result.reason)
    return result
