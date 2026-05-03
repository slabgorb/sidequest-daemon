"""Regression test for playtest 2026-04-30 — daemon EOF on compose error.

Before this fix, any compose-time exception (`RenderConfigError`,
`StyleMissError`, `CatalogMissError`, `BudgetError`, the `ValueError`
that `PlaceCatalog.get` raises on a non-`where:` ref) bubbled out of
`_handle_client` because the outer try only caught
`ConnectionResetError, BrokenPipeError`. The connection closed mid-
request, the server saw `eof_before_reply`, and the failure surfaced as
`daemon_unavailable` — masking the real reason behind a transport
error.

These tests verify the JSON-RPC contract: every `render` request must
get either a `result` or an `error` frame back, never an EOF.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from sidequest_daemon.media.daemon import _handle_client


class _RecordingWriter:
    """Minimal stand-in for `asyncio.StreamWriter` that records the
    bytes the handler tries to send. Implements only the methods
    `_handle_client` and `_write` actually call.
    """

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self._closed = False

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        # Heartbeat emission (story 45-31) calls drain() after writing
        # the per-queue heartbeat line on connection accept. The real
        # StreamWriter blocks until the kernel buffer drains; in the
        # in-memory recorder we just track that drain was called.
        return None

    def get_extra_info(self, key: str) -> str:
        return "test-peer"

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        return None

    @property
    def replies(self) -> list[dict]:
        """Return JSON-RPC reply frames only. Story 45-31 added per-queue
        heartbeat lines on connection accept that share the wire with
        replies; filter them out so existing tests reason about replies
        in isolation."""
        joined = b"".join(self.chunks).decode()
        out: list[dict] = []
        for line in joined.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("event") == "heartbeat":
                continue
            out.append(obj)
        return out


@pytest.mark.asyncio
async def test_render_missing_required_field_replies_compose_failed_not_eof():
    """A render request with a missing required field (subject / world /
    genre) used to crash the connection via uncaught
    `RenderConfigError`. It must now reply with a structured
    `COMPOSE_FAILED` JSON-RPC error.
    """
    reader = asyncio.StreamReader()
    request = (
        json.dumps(
            {
                "id": "test-missing-world",
                "method": "render",
                # Deliberately omit `world` so the daemon's pre-flight
                # gate raises RenderConfigError. `subject` and `genre`
                # are present so we exercise the missing-field branch.
                "params": {
                    "tier": "scene_illustration",
                    "subject": "test scene",
                    "genre": "testgenre",
                    # No `narration` — skip the SceneInterpreter branch
                    # so the error surfaces from the compose gate, not
                    # from the extractor.
                },
            }
        )
        + "\n"
    ).encode()
    reader.feed_data(request)
    reader.feed_eof()

    writer = _RecordingWriter()

    # Stub pool — should not be touched on a compose-time failure.
    class _UnusedPool:
        def render(self, params: dict) -> dict:
            raise AssertionError(
                "pool.render must NOT be called when compose pre-flight fails"
            )

        def status(self) -> dict:
            return {}

    await _handle_client(
        reader,
        writer,  # type: ignore[arg-type]
        _UnusedPool(),  # type: ignore[arg-type]
        asyncio.Lock(),
        asyncio.Lock(),
    )

    replies = writer.replies
    assert len(replies) == 1, (
        f"expected exactly one reply (no EOF, no extra frames), got {replies}"
    )
    reply = replies[0]
    assert reply["id"] == "test-missing-world"
    assert "error" in reply, (
        f"expected JSON-RPC error frame, not result. Got {reply}"
    )
    err = reply["error"]
    assert err["code"] == "COMPOSE_FAILED"
    assert err["error_type"] == "RenderConfigError"
    assert err["tier"] == "scene_illustration"


@pytest.mark.asyncio
async def test_render_value_error_from_place_catalog_replies_compose_failed():
    """The 2026-04-30 playtest signature: server sends `location` as
    free-form prose ("Engine Bay"), `PlaceCatalog.get` raises
    `ValueError("place ref 'Engine Bay' must use scheme 'where:'")`,
    daemon used to EOF. Must now reply with `COMPOSE_FAILED`.

    We force the path by stubbing `compose_prompt_for` to raise the same
    ValueError the catalog would — the test stays unit-scoped (no genre
    pack fixtures, no MLX) while still covering the real code path that
    was missing the catch.
    """
    from sidequest_daemon.media.workers import zimage_mlx_worker

    reader = asyncio.StreamReader()
    request = (
        json.dumps(
            {
                "id": "test-place-ref-violation",
                "method": "render",
                "params": {
                    "tier": "scene_illustration",
                    "subject": "Sprung exploration locker in red corridor light",
                    "genre": "space_opera",
                    "world": "coyote_star",
                    "location": "Engine Bay",  # the server contract bug
                },
            }
        )
        + "\n"
    ).encode()
    reader.feed_data(request)
    reader.feed_eof()

    writer = _RecordingWriter()

    def _raise_place_violation(cue):
        raise ValueError("place ref 'Engine Bay' must use scheme 'where:'")

    original_compose = zimage_mlx_worker.compose_prompt_for
    zimage_mlx_worker.compose_prompt_for = _raise_place_violation
    try:
        class _UnusedPool:
            def render(self, params: dict) -> dict:
                raise AssertionError(
                    "pool.render must NOT be called when compose raises"
                )

            def status(self) -> dict:
                return {}

        await _handle_client(
            reader,
            writer,  # type: ignore[arg-type]
            _UnusedPool(),  # type: ignore[arg-type]
            asyncio.Lock(),
            asyncio.Lock(),
        )
    finally:
        zimage_mlx_worker.compose_prompt_for = original_compose

    replies = writer.replies
    assert len(replies) == 1, f"expected exactly one reply, got {replies}"
    reply = replies[0]
    assert reply["id"] == "test-place-ref-violation"
    assert "error" in reply
    err = reply["error"]
    assert err["code"] == "COMPOSE_FAILED"
    assert err["error_type"] == "ValueError"
    assert "where:" in err["message"]
