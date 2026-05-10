"""MusicPipeline — orchestrates one ACE-Step generation job end-to-end.

Reads a per-track JSON params file, derives the R2 destination key from
the file's path, runs the adapter, converts WAV → OGG, uploads to R2,
emits watcher events at every stage. Per spec
docs/superpowers/specs/2026-05-09-daemon-between-session-music-generation-design.md.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_GENRE_PACKS_RE = re.compile(r".*?(genre_packs/.*?)/audio/music/(.+?)_input_params\.json$")


@dataclass
class MusicResult:
    r2_key: str
    duration_ms: int
    seed: int
    elapsed_ms: int


class MusicPipeline:
    """Single-job orchestrator. Constructed once per daemon process,
    reused across requests."""

    def __init__(self, *, adapter, r2_uploader, watcher, render_lock):
        self._adapter = adapter
        self._r2_uploader = r2_uploader
        self._watcher = watcher
        self._render_lock = render_lock

    @staticmethod
    def derive_r2_key(json_path: Path) -> str:
        """Strip `_input_params.json`, append `.ogg`, anchor under
        `genre_packs/<pack>/`. Raises ValueError if path doesn't fit."""
        s = str(json_path)
        m = _GENRE_PACKS_RE.match(s)
        if not m:
            raise ValueError(
                f"INVALID_PARAMS_LOCATION: {json_path} is not under a "
                f"genre_packs/<pack>/audio/music/ directory or is missing "
                f"the _input_params.json suffix"
            )
        pack_path, name = m.group(1), m.group(2)
        return f"{pack_path}/audio/music/{name}.ogg"
