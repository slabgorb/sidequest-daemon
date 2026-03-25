"""Configuration helpers for the sidequest-daemon."""

from __future__ import annotations

import os
from pathlib import Path


def genre_packs_path() -> Path:
    """Return the genre packs directory.

    Resolution order:
    1. SIDEQUEST_GENRE_PACKS env var
    2. ../genre_packs relative to this repo (orchestrator sibling convention)
    3. ./genre_packs in cwd
    """
    env = os.environ.get("SIDEQUEST_GENRE_PACKS")
    if env:
        return Path(env)

    # When run from the orc-quest orchestrator layout:
    #   orc-quest/sidequest-daemon/sidequest_daemon/config.py
    #   → orc-quest/genre_packs/
    repo_sibling = Path(__file__).parent.parent.parent / "genre_packs"
    if repo_sibling.is_dir():
        return repo_sibling

    return Path.cwd() / "genre_packs"
