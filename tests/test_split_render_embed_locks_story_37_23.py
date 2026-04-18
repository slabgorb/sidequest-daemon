"""Wiring + behavior tests for story 37-23: split render_lock into render_lock + embed_lock.

Prior state (story 37-5): embed and Flux shared a single ``render_lock``. The shared
lock existed to prevent a concurrent-MPS-model-session deadlock where a cold-load
embed racing a Flux render on the same Metal device hung the driver.

This story's architectural decision (see SM assessment + design deviation in the
session file): **move embed off MPS entirely.** ``SentenceTransformer`` runs on CPU,
giving embed a device that Flux never touches. With independent devices, the locks
become independent: ``render_lock`` guards Flux/MPS, ``embed_lock`` guards embed/CPU.

An embed request issued while Flux holds ``render_lock`` must complete within embed
latency budget — that's the whole point of the refactor.

Regression guards follow the 37-5 pattern: source-level inspection + a real-async
concurrency proof. No daemon process required — pure unit-level assertions.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from pathlib import Path
from typing import Any

import pytest

from sidequest_daemon.media import daemon as daemon_mod
from sidequest_daemon.media.daemon import EmbedWorker, WorkerPool, _handle_client, _run_daemon


DAEMON_SOURCE = Path(inspect.getsourcefile(daemon_mod)).read_text()


# ============================================================
# Invariant 1 — EmbedWorker pins to CPU, never MPS
# ============================================================


class TestEmbedWorkerOnCpu:
    """The embed model MUST load on CPU so it never races Flux on MPS.

    This is the whole architectural trade of story 37-23. If a future refactor
    moves embed back to MPS without also re-introducing the shared lock, the
    2026-04-10 deadlock returns.
    """

    def test_load_model_forces_cpu_device(self):
        """``_load_model`` must pass ``device="cpu"`` to SentenceTransformer."""
        src = inspect.getsource(EmbedWorker._load_model)
        # Accept either keyword form — ``device="cpu"`` or ``device='cpu'``
        assert re.search(r'device\s*=\s*["\']cpu["\']', src), (
            "EmbedWorker._load_model must construct SentenceTransformer with "
            'device="cpu" — embed runs on CPU so it never contends with Flux '
            "on the MPS device (story 37-23 architectural invariant)"
        )

    def test_load_model_does_not_reference_mps(self):
        """No ``device='mps'`` string in the embed load path."""
        src = inspect.getsource(EmbedWorker._load_model)
        assert not re.search(r'device\s*=\s*["\']mps["\']', src), (
            "EmbedWorker._load_model must not pin device to MPS — embed is a "
            "CPU-only worker as of story 37-23"
        )


# ============================================================
# Invariant 2 — _run_daemon constructs two distinct locks
# ============================================================


class TestDaemonConstructsBothLocks:
    """The production entry point must create render_lock AND embed_lock."""

    def test_run_daemon_source_creates_render_lock(self):
        src = inspect.getsource(_run_daemon)
        assert re.search(r"render_lock\s*=\s*asyncio\.Lock\s*\(\s*\)", src), (
            "_run_daemon must instantiate render_lock = asyncio.Lock() — the "
            "production wiring point for Flux serialization"
        )

    def test_run_daemon_source_creates_embed_lock(self):
        src = inspect.getsource(_run_daemon)
        assert re.search(r"embed_lock\s*=\s*asyncio\.Lock\s*\(\s*\)", src), (
            "_run_daemon must instantiate embed_lock = asyncio.Lock() — the "
            "production wiring point for embed serialization (story 37-23)"
        )

    def test_run_daemon_passes_embed_lock_to_handler(self):
        """The handler factory closure must forward embed_lock.

        Guards against the 'added the field but didn't wire it' failure mode.
        """
        src = inspect.getsource(_run_daemon)
        # The handler is registered as: lambda r, w: _handle_client(r, w, pool, render_lock, embed_lock)
        # Match the call with both locks as args — order-sensitive by signature.
        pattern = re.compile(
            r"_handle_client\s*\([^)]*\brender_lock\b[^)]*\bembed_lock\b[^)]*\)",
            re.DOTALL,
        )
        assert pattern.search(src), (
            "_run_daemon must pass BOTH render_lock and embed_lock to "
            "_handle_client — otherwise the lock is defined but not reachable "
            "from the production request path"
        )


# ============================================================
# Invariant 3 — _handle_client signature accepts both locks
# ============================================================


class TestHandlerSignature:
    """_handle_client's parameter list must name both locks explicitly."""

    def test_handle_client_accepts_embed_lock_parameter(self):
        sig = inspect.signature(_handle_client)
        assert "embed_lock" in sig.parameters, (
            "_handle_client must accept embed_lock as a named parameter — "
            "story 37-23 splits the lock at the dispatcher level"
        )

    def test_handle_client_accepts_render_lock_parameter(self):
        sig = inspect.signature(_handle_client)
        assert "render_lock" in sig.parameters, (
            "_handle_client must still accept render_lock — Flux renders "
            "continue to serialize on the MPS device"
        )

    def test_embed_lock_is_asyncio_lock_annotated(self):
        sig = inspect.signature(_handle_client)
        param = sig.parameters["embed_lock"]
        # The annotation should be asyncio.Lock. Accept str form for forward refs.
        anno = param.annotation
        anno_str = getattr(anno, "__qualname__", None) or str(anno)
        assert "Lock" in anno_str, (
            f"embed_lock parameter must be annotated asyncio.Lock, got {anno_str!r}"
        )


# ============================================================
# Invariant 4 — embed handler uses embed_lock (not render_lock)
# ============================================================


def _embed_handler_block() -> str:
    """Extract the source of the ``elif method == "embed":`` branch."""
    marker = 'elif method == "embed":'
    start = DAEMON_SOURCE.index(marker)
    end = DAEMON_SOURCE.index("else:", start + len(marker))
    return DAEMON_SOURCE[start:end]


def _render_handler_block() -> str:
    """Extract the source of the ``if method == "render":`` branch (first method).

    Bounded by the next ``elif method == ...`` marker.
    """
    marker = 'if method == "render":'
    start = DAEMON_SOURCE.index(marker)
    end = DAEMON_SOURCE.index("elif method ==", start + len(marker))
    return DAEMON_SOURCE[start:end]


class TestEmbedHandlerUsesEmbedLock:
    def test_embed_handler_acquires_embed_lock(self):
        block = _embed_handler_block()
        # Strip comments so documentation references don't satisfy the check.
        code_only = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        )
        assert "async with embed_lock:" in code_only, (
            "embed handler must acquire embed_lock — story 37-23 splits the "
            "lock so embed no longer blocks behind Flux renders"
        )

    def test_embed_handler_does_not_acquire_render_lock(self):
        """Embed must NOT acquire render_lock — that's the old serialization pattern."""
        block = _embed_handler_block()
        code_only = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        )
        assert "async with render_lock:" not in code_only, (
            "embed handler must not acquire render_lock — that re-introduces "
            "the contention tax story 37-23 was created to remove"
        )


# ============================================================
# Invariant 5 — render handler uses render_lock (not embed_lock)
# ============================================================


class TestRenderHandlerUsesRenderLock:
    def test_render_handler_acquires_render_lock(self):
        block = _render_handler_block()
        code_only = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        )
        assert "async with render_lock:" in code_only, (
            "render handler must acquire render_lock — Flux/MPS serialization "
            "is still required"
        )

    def test_render_handler_does_not_acquire_embed_lock(self):
        block = _render_handler_block()
        code_only = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        )
        assert "async with embed_lock:" not in code_only, (
            "render handler must not acquire embed_lock — only the embed path "
            "owns that lock (story 37-23)"
        )


# ============================================================
# Invariant 6 — no call site holds BOTH locks simultaneously (deadlock guard)
# ============================================================


class TestNoNestedLockAcquisition:
    """No nested ``async with`` over both locks anywhere in the daemon source.

    If any future code path acquires render_lock then embed_lock (or vice
    versa) within the same scope, lock-ordering inversion elsewhere could
    produce a real deadlock. Block the pattern at review time.

    Implementation note: uses AST walking (not regex) to distinguish true
    nesting from sibling ``elif`` branches. Two ``async with`` blocks in
    separate ``elif`` arms of the same function are NOT nested.
    """

    @staticmethod
    def _find_nested_lock_pairs(
        outer_lock: str, inner_lock: str
    ) -> list[tuple[int, int]]:
        """Walk the daemon AST and return (outer_line, inner_line) pairs where
        ``inner_lock`` is acquired inside the body of ``outer_lock``'s
        ``async with`` block — true nesting, not sibling branches."""
        import ast

        tree = ast.parse(DAEMON_SOURCE)
        findings: list[tuple[int, int]] = []

        def _references_lock(node: ast.AST, lock_name: str) -> bool:
            """True if node is an ``async with <lock_name>:`` statement."""
            if not isinstance(node, ast.AsyncWith):
                return False
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Name) and ctx.id == lock_name:
                    return True
            return False

        class _NestedLockVisitor(ast.NodeVisitor):
            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                if _references_lock(node, outer_lock):
                    for inner in ast.walk(node):
                        if inner is node:
                            continue
                        if _references_lock(inner, inner_lock):
                            findings.append((node.lineno, inner.lineno))
                self.generic_visit(node)

        _NestedLockVisitor().visit(tree)
        return findings

    def test_no_nested_lock_acquire_render_then_embed(self):
        pairs = self._find_nested_lock_pairs("render_lock", "embed_lock")
        assert not pairs, (
            "Nested acquisition detected: render_lock held while acquiring "
            f"embed_lock at line(s) {pairs}. Lock-ordering inversion risk — "
            "no code path may hold both locks simultaneously."
        )

    def test_no_nested_lock_acquire_embed_then_render(self):
        pairs = self._find_nested_lock_pairs("embed_lock", "render_lock")
        assert not pairs, (
            "Nested acquisition detected: embed_lock held while acquiring "
            f"render_lock at line(s) {pairs}. Lock-ordering inversion risk."
        )


# ============================================================
# Invariant 7 — concurrency proof: embed does not block on render
# ============================================================


class _FakePool:
    """Minimal pool stand-in for concurrency tests.

    ``render`` sleeps to simulate a long Flux render; ``embed`` returns a
    tiny vector immediately. Mirrors the real WorkerPool surface used by
    _handle_client without loading any models.
    """

    def __init__(self, render_sleep_s: float) -> None:
        self._render_sleep_s = render_sleep_s
        self.render_calls = 0
        self.embed_calls = 0

    def render(self, params: dict) -> dict:
        self.render_calls += 1
        time.sleep(self._render_sleep_s)
        return {"path": "/tmp/fake.png", "tier": params.get("tier")}

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [0.0, 0.0, 0.0]


class _StubWriter:
    """Captures written frames for assertion."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.frames.append(data)

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name: str) -> Any:
        return "test-peer"

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


async def _feed_reader(lines: list[bytes]) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with newline-terminated JSON frames."""
    reader = asyncio.StreamReader()
    for line in lines:
        reader.feed_data(line)
    reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_embed_does_not_block_on_in_flight_render():
    """The core behavioral guarantee of story 37-23.

    Simulate: render is holding render_lock (long sleep). Meanwhile an embed
    request arrives. Because the locks are independent, the embed handler
    should acquire embed_lock immediately and return within embed latency
    budget — NOT wait for the render to finish.

    Pre-fix (single shared lock): embed waits ~render_sleep_s for the render
    to complete, then runs. Post-fix: embed returns promptly.
    """
    import json

    render_sleep_s = 0.5  # simulated long Flux render
    embed_budget_s = 0.2  # must complete well inside the render window

    pool = _FakePool(render_sleep_s=render_sleep_s)
    render_lock = asyncio.Lock()
    embed_lock = asyncio.Lock()

    # Task A: issues a render request — this will hold render_lock for ~0.5s.
    render_reader = await _feed_reader(
        [json.dumps({"id": "r1", "method": "render", "params": {"tier": "portrait"}}).encode() + b"\n"]
    )
    render_writer = _StubWriter()

    # Task B: issues an embed request — should NOT wait for render.
    embed_reader = await _feed_reader(
        [json.dumps({"id": "e1", "method": "embed", "params": {"text": "hello"}}).encode() + b"\n"]
    )
    embed_writer = _StubWriter()

    # Kick off the render first so it holds render_lock before embed starts.
    render_task = asyncio.create_task(
        _handle_client(render_reader, render_writer, pool, render_lock, embed_lock)
    )
    # Tiny yield to let the render task claim render_lock.
    await asyncio.sleep(0.05)
    assert render_lock.locked(), (
        "Precondition failed: render task did not acquire render_lock in time — "
        "test timing needs adjustment"
    )

    embed_start = time.monotonic()
    await _handle_client(embed_reader, embed_writer, pool, render_lock, embed_lock)
    embed_elapsed = time.monotonic() - embed_start

    await render_task  # let the render finish cleanly

    assert embed_elapsed < embed_budget_s, (
        f"Embed request blocked on render: took {embed_elapsed * 1000:.0f}ms, "
        f"budget {embed_budget_s * 1000:.0f}ms. With independent locks, embed "
        f"must not wait for Flux renders. Shared-lock regression likely."
    )
    assert pool.embed_calls == 1, "embed handler did not invoke pool.embed"


# ============================================================
# Comment hygiene — the stale 37-5 comment must be updated
# ============================================================


class TestStaleCommentRemoved:
    """The pool.embed docstring referenced the shared render_lock — must update."""

    def test_pool_embed_docstring_no_longer_says_acquire_render_lock(self):
        doc = inspect.getdoc(WorkerPool.embed) or ""
        assert "render_lock" not in doc, (
            "WorkerPool.embed docstring still references render_lock — "
            "story 37-23 moved embed off MPS. Update the docstring to "
            "describe embed_lock + CPU device invariant."
        )
