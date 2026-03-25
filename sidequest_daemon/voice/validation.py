"""Startup validation for required voice models (Story 35-4).

Checks genre pack required_voice_models against available engines,
logging warnings for missing models without crashing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidequest_daemon.genre.models import GenrePack
    from sidequest_daemon.voice.protocol import SynthesisEngine

logger = logging.getLogger(__name__)


def validate_required_voice_models(
    genre_pack: GenrePack,
    available_engines: list[SynthesisEngine],
) -> list[str]:
    """Validate that required voice models are available.

    Args:
        genre_pack: The loaded genre pack with required_voice_models field.
        available_engines: Currently available synthesis engines.

    Returns:
        List of missing model names (empty if all available).
    """
    required = genre_pack.required_voice_models
    if not required:
        return []

    available_names = {engine.name.lower() for engine in available_engines}
    missing = [
        model for model in required if model.lower() not in available_names
    ]

    for model in missing:
        logger.warning(
            "Required voice model '%s' is not available. "
            "Voice synthesis may fall back to another engine.",
            model,
        )

    return missing
