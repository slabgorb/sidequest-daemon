"""Regression test for playtest 2026-04-30 — daemon was silently
overriding the server-supplied tier classification.

Pre-fix flow (the bug):
1. Server narrator agent emitted a structured visual block:
   ``{tier: "landscape", subject: "Docking gantry...", ...}``.
2. Server forwarded those fields to the daemon along with the raw
   ``narration`` prose.
3. Daemon ran ``SceneInterpreter`` on the narration anyway, matched an
   atmosphere rule, overwrote ``params["tier"]`` to
   ``"scene_illustration"``.
4. ``build_render_target`` then constructed
   ``RenderTarget(kind="illustration", participants=[], ...)`` because
   the server only populates ``params["characters"]`` on its own
   ``tier=="scene_illustration"`` branch — which the server hadn't
   taken (it had chosen ``landscape``).
5. The pydantic validator raised
   ``illustration targets require participants`` and the daemon
   replied ``COMPOSE_FAILED``.

Post-fix: when the caller already supplied a usable structured visual
block (``params["tier"]`` in IMAGE_TIERS *and* a non-empty
``params["subject"]``), SceneInterpreter is skipped — the narrator's
classification is authoritative. Document extraction still runs
unconditionally because that is a separate concern (text overlays).
"""

from __future__ import annotations

import asyncio
import json

import pytest


class _RecordingWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    def get_extra_info(self, key: str) -> str:
        return "test-peer"

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        return None

    @property
    def replies(self) -> list[dict]:
        joined = b"".join(self.chunks).decode()
        return [
            json.loads(line) for line in joined.splitlines() if line.strip()
        ]


class _UnusedPool:
    def render(self, params: dict) -> dict:
        raise AssertionError(
            "pool.render must not be called when compose preempts dispatch"
        )

    def status(self) -> dict:
        return {}


async def _drive_handler(reader, writer):
    from sidequest_daemon.media.daemon import _handle_client

    await _handle_client(
        reader,
        writer,
        _UnusedPool(),
        asyncio.Lock(),
        asyncio.Lock(),
    )


@pytest.mark.asyncio
async def test_server_supplied_tier_and_subject_skips_scene_interpreter():
    """Reproducer for the playtest 2026-04-30 Parsley turn-1
    COMPOSE_FAILED:

    The server sent ``tier=landscape`` with a non-empty ``subject``.
    The daemon must NOT rewrite tier to ``scene_illustration`` via
    SceneInterpreter — the narrator agent already classified.
    """
    from sidequest_daemon.media.workers import zimage_mlx_worker

    captured: dict[str, object] = {}

    def _capture(cue):
        captured["tier"] = cue.tier
        captured["subject"] = cue.subject
        # Raise so we don't need a fully-loaded composer pipeline; the
        # error frame is fine for this test, what we're proving is the
        # CUE that arrived at compose_prompt_for.
        raise ValueError("intentional compose-stop after capture")

    original_compose = zimage_mlx_worker.compose_prompt_for
    zimage_mlx_worker.compose_prompt_for = _capture
    try:
        reader = asyncio.StreamReader()
        request = (
            json.dumps(
                {
                    "id": "scene-interp-skip-server-tier",
                    "method": "render",
                    "params": {
                        "tier": "landscape",
                        "subject": "Docking gantry of welded station crescent in red strobe-light, gas giant filling",
                        "mood": "tense",
                        "tags": [],
                        "genre": "space_opera",
                        "world": "coyote_reach",
                        # Narration that, pre-fix, would trigger an
                        # atmosphere rule match and rewrite tier to
                        # scene_illustration. The fix keeps server's
                        # tier authoritative.
                        "narration": (
                            "The clamps shudder, half-seated. "
                            "Through the viewport, gantry lights "
                            "strobe red across three welded hulls."
                        ),
                    },
                }
            )
            + "\n"
        ).encode()
        reader.feed_data(request)
        reader.feed_eof()

        writer = _RecordingWriter()
        await _drive_handler(reader, writer)
    finally:
        zimage_mlx_worker.compose_prompt_for = original_compose

    from sidequest_daemon.renderer.models import RenderTier

    assert captured.get("tier") == RenderTier.LANDSCAPE, (
        f"expected server's LANDSCAPE tier to survive into compose, got "
        f"{captured.get('tier')!r}. SceneInterpreter is overriding the "
        f"narrator's authoritative classification — the playtest "
        f"2026-04-30 COMPOSE_FAILED regression."
    )
    assert "Docking gantry" in str(captured.get("subject", "")), (
        f"server-supplied subject must survive, got {captured.get('subject')!r}"
    )


@pytest.mark.asyncio
async def test_no_server_tier_still_runs_scene_interpreter():
    """When the server does not supply a structured visual block (no
    tier or empty subject), SceneInterpreter should still run as a
    fallback. The skip is conditional on the server actually having
    classified.
    """
    from sidequest_daemon.media.workers import zimage_mlx_worker

    captured: dict[str, object] = {}

    def _capture(cue):
        captured["tier"] = cue.tier
        captured["subject"] = cue.subject
        raise ValueError("intentional compose-stop after capture")

    original_compose = zimage_mlx_worker.compose_prompt_for
    zimage_mlx_worker.compose_prompt_for = _capture
    try:
        reader = asyncio.StreamReader()
        request = (
            json.dumps(
                {
                    "id": "scene-interp-runs-without-tier",
                    "method": "render",
                    "params": {
                        # Empty subject and no tier → fallback path.
                        "subject": "",
                        "tier": "",
                        "genre": "space_opera",
                        "world": "coyote_reach",
                        "narration": (
                            "The dust settles. A generator coughs and "
                            "catches. Far Landing wakes around you."
                        ),
                    },
                }
            )
            + "\n"
        ).encode()
        reader.feed_data(request)
        reader.feed_eof()

        writer = _RecordingWriter()
        await _drive_handler(reader, writer)
    finally:
        zimage_mlx_worker.compose_prompt_for = original_compose

    # SceneInterpreter is allowed to pick any tier here — what matters is
    # that compose was reached (it captured), proving the fallback
    # pipeline still fires when the server didn't classify.
    assert "tier" in captured, (
        "SceneInterpreter should still populate tier when the server "
        "supplied no structured visual block. Got captured=%r" % captured
    )
