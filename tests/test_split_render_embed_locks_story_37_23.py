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

No daemon subprocess required — tests combine source-level inspection, AST walking,
a behavioral mock of ``SentenceTransformer`` for the CPU invariant, a real asyncio
concurrency proof with modeled embed latency, and a negative-regression test that
proves the concurrency harness CAN detect a shared-lock reversion (ensuring the
green path is not a tautology).
"""

from __future__ import annotations

import asyncio
import inspect
import re
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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

    def test_load_model_constructs_sentence_transformer_with_cpu_device(self):
        """Behavioral proof (refactor-resistant): mock SentenceTransformer and
        assert _load_model invokes it with ``device="cpu"`` regardless of how
        the method is internally structured.

        The source-grep tests above are brittle under refactoring — if
        ``_load_model`` is ever split into a helper like
        ``_build_model(device="cpu")``, the source check passes vacuously
        because the kwarg no longer appears directly in ``_load_model``'s
        body. This test mocks the constructor at the import path the
        production code uses and asserts on the actual call, catching any
        internal refactor that loses the CPU pin.
        """
        import sys
        # Ensure a fresh model load — clear any cached singleton state
        # that might skip the constructor call.
        worker = EmbedWorker()

        fake_model = MagicMock(name="SentenceTransformerInstance")
        fake_constructor = MagicMock(return_value=fake_model)

        # sentence_transformers is imported inside _load_model (lazy import)
        # so we patch at the module level where it will be looked up.
        fake_module = MagicMock()
        fake_module.SentenceTransformer = fake_constructor

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            result = worker._load_model()

        assert fake_constructor.called, (
            "_load_model did not call SentenceTransformer — refactor may have "
            "lost the model construction call entirely"
        )
        call_kwargs = fake_constructor.call_args.kwargs
        assert call_kwargs.get("device") == "cpu", (
            f"_load_model constructed SentenceTransformer with "
            f"device={call_kwargs.get('device')!r}, expected 'cpu'. "
            f"Story 37-23 invariant: embed never runs on MPS."
        )
        assert result is fake_model, (
            "_load_model must return the SentenceTransformer instance it built"
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
    """Pool stand-in for concurrency tests.

    Both ``render`` and ``embed`` sleep to model real latencies. Modeling embed
    latency is critical: with a zero-cost fake, ``asyncio.to_thread`` returns
    almost immediately and the test would pass even with a shared lock — a
    tautology. A realistic embed latency makes the concurrency proof actually
    discriminate between independent-lock and shared-lock designs.

    A ``threading.Event`` is set as soon as ``render`` acquires its work
    (i.e. the sync body begins executing); test code can await the event
    instead of relying on ``asyncio.sleep`` timing to synchronize with the
    thread pool.
    """

    def __init__(
        self,
        render_sleep_s: float = 0.5,
        embed_sleep_s: float = 0.03,
    ) -> None:
        self._render_sleep_s = render_sleep_s
        self._embed_sleep_s = embed_sleep_s
        self.render_calls = 0
        self.embed_calls = 0
        # Set by render() the instant the sync body starts — after
        # asyncio.to_thread has dispatched into the pool and the worker
        # thread is running. Test code awaits this to eliminate timing
        # races on render_lock acquisition.
        self.render_started = threading.Event()

    def render(self, params: dict) -> dict:
        self.render_calls += 1
        # Signal BEFORE sleeping so the embed task can proceed as soon as
        # the render body is running (and, importantly, render_lock is held
        # by the calling async context before this sync body was dispatched).
        self.render_started.set()
        time.sleep(self._render_sleep_s)
        return {"path": "/tmp/fake.png", "tier": params.get("tier")}

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        # Model realistic CPU embed latency (~30ms). Without this the
        # concurrency test is a tautology: asyncio.to_thread + zero-cost
        # work returns so fast that a shared lock would also appear to
        # pass. See `test_concurrency_harness_detects_shared_lock_regression`.
        time.sleep(self._embed_sleep_s)
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


async def _run_render_then_embed(
    pool: _FakePool,
    render_lock: asyncio.Lock,
    embed_lock: asyncio.Lock,
) -> float:
    """Run a render concurrent with an embed; return embed elapsed seconds.

    Shared helper so the positive case (independent locks) and the negative
    regression case (shared lock) drive the exact same harness. The only
    variable between them is whether render_lock and embed_lock are
    distinct objects.
    """
    import json

    render_reader = await _feed_reader(
        [json.dumps({"id": "r1", "method": "render", "params": {"tier": "portrait", "positive_prompt": "x"}}).encode() + b"\n"]
    )
    render_writer = _StubWriter()
    embed_reader = await _feed_reader(
        [json.dumps({"id": "e1", "method": "embed", "params": {"text": "hello"}}).encode() + b"\n"]
    )
    embed_writer = _StubWriter()

    # Kick off the render first. render_lock is acquired by _handle_client
    # BEFORE it dispatches to asyncio.to_thread; pool.render.render_started
    # fires once the sync body is running in the thread pool. Awaiting the
    # event guarantees render_lock is held before we measure embed.
    render_task = asyncio.create_task(
        _handle_client(render_reader, render_writer, pool, render_lock, embed_lock)
    )

    # Event-based synchronization — no timing assumption.
    started = asyncio.get_event_loop().run_in_executor(
        None, pool.render_started.wait, 2.0  # 2s safety cap
    )
    acquired = await started
    assert acquired, (
        "Render task did not begin within 2s — thread pool exhaustion or "
        "deadlock prevented _handle_client from dispatching render"
    )
    assert render_lock.locked(), (
        "render_started fired but render_lock is not held — handler logic "
        "changed and no longer acquires render_lock before dispatching work"
    )

    embed_start = time.monotonic()
    await _handle_client(embed_reader, embed_writer, pool, render_lock, embed_lock)
    embed_elapsed = time.monotonic() - embed_start

    await render_task
    return embed_elapsed


@pytest.mark.asyncio
async def test_embed_does_not_block_on_in_flight_render():
    """The core behavioral guarantee of story 37-23 (positive case).

    Render holds render_lock; embed uses a SEPARATE embed_lock. Embed must
    complete proportional to its own modeled latency, not wait for the
    render to finish.

    Budget is proportional: embed must finish in under half the render
    window. That's an order-of-magnitude separation the lock architecture
    must deliver; it is not a tight wall-clock budget that would flake on
    a loaded CI runner.
    """
    pool = _FakePool(render_sleep_s=0.5, embed_sleep_s=0.03)
    render_lock = asyncio.Lock()
    embed_lock = asyncio.Lock()

    embed_elapsed = await _run_render_then_embed(pool, render_lock, embed_lock)

    # Proportional budget: embed must finish in under half the render time.
    # With independent locks, embed runs in ~30ms; render sleeps 500ms.
    # A shared-lock regression would push embed past 500ms. A budget of
    # render_sleep_s/2 = 250ms gives a 5x margin over the true 30ms cost
    # while still firmly catching any regression that serializes embed
    # behind render.
    proportional_budget_s = 0.5 / 2
    assert embed_elapsed < proportional_budget_s, (
        f"Embed request blocked on render: took {embed_elapsed * 1000:.0f}ms, "
        f"budget {proportional_budget_s * 1000:.0f}ms (= render_sleep/2). "
        f"With independent locks, embed must not wait for Flux renders. "
        f"Shared-lock regression likely."
    )
    assert pool.embed_calls == 1, "embed handler did not invoke pool.embed"
    assert pool.render_calls == 1, "render handler did not invoke pool.render"


@pytest.mark.asyncio
async def test_concurrency_harness_detects_shared_lock_regression():
    """Negative-regression proof: the harness above actually catches the bug.

    Runs the SAME concurrency scenario but passes ONE lock as both
    render_lock and embed_lock — simulating a regression where someone
    reverts 37-23 and re-shares the locks. Embed must now serialize behind
    render, and the elapsed time should exceed the positive case's budget.

    If this test ever starts passing (i.e. shared-lock also satisfies the
    <budget assertion), the positive test is tautological and both are
    broken. This negative test is the backstop that proves the positive
    test is a real detector, not a pass-through.
    """
    pool = _FakePool(render_sleep_s=0.5, embed_sleep_s=0.03)
    shared_lock = asyncio.Lock()

    embed_elapsed = await _run_render_then_embed(pool, shared_lock, shared_lock)

    # With a shared lock, embed must wait for render to finish — roughly
    # the render_sleep_s (0.5s) MINUS however much render had already slept
    # when embed arrived. Conservatively: embed must take longer than the
    # proportional budget used in the positive test. If it doesn't, the
    # harness isn't actually serializing work, and the positive test is
    # tautological.
    proportional_budget_s = 0.5 / 2
    assert embed_elapsed >= proportional_budget_s, (
        f"Harness did not detect shared-lock regression: embed took "
        f"{embed_elapsed * 1000:.0f}ms under a shared lock but the "
        f"positive-test budget is {proportional_budget_s * 1000:.0f}ms. "
        f"This means _FakePool.embed is too cheap to exercise the lock, or "
        f"_handle_client dispatches in a way that doesn't serialize through "
        f"the shared lock. The positive test would pass vacuously."
    )


# ============================================================
# Comment hygiene — the stale 37-5 comment must be updated
# ============================================================


class TestOtelInstrumentation:
    """Per CLAUDE.md: every subsystem fix MUST emit OTEL spans so the GM panel
    can verify the fix is engaged. Story 37-23 changes concurrent-dispatch
    behavior — a prime candidate for silent regression (embed re-serializing
    behind render) that only an OTEL span can detect at runtime.

    Idiom (matches existing daemon OTEL, e.g. flux_mlx_worker.py:86):
        tracer = trace.get_tracer("sidequest_daemon.media.daemon")
        with tracer.start_as_current_span("daemon.dispatch.embed") as span:
            span.set_attribute("lock_name", "embed_lock")
            ...

    These tests verify the span emission structurally. The GM panel can then
    surface the spans via the ADR-058 Claude-subprocess OTEL passthrough.
    """

    def test_daemon_module_imports_opentelemetry_trace(self):
        """The daemon module must import the OTEL trace API."""
        has_trace_import = (
            re.search(r"from\s+opentelemetry\s+import\s+.*trace", DAEMON_SOURCE)
            or re.search(r"import\s+opentelemetry\.trace", DAEMON_SOURCE)
        )
        assert has_trace_import, (
            "sidequest_daemon/media/daemon.py must import opentelemetry.trace "
            "to emit dispatch spans. CLAUDE.md OTEL obligation: every "
            "subsystem fix must be GM-panel-visible. See flux_mlx_worker.py "
            "for the canonical import pattern."
        )

    def test_embed_dispatch_opens_otel_span(self):
        """Embed branch must open an OTEL span naming the dispatch operation."""
        block = _embed_handler_block()
        # Accept either start_as_current_span or start_span — the flux worker
        # uses start_as_current_span, which is the preferred pattern.
        span_pattern = re.compile(
            r"start_as_current_span\s*\(\s*[\"'][^\"']*embed[^\"']*[\"']",
            re.IGNORECASE,
        )
        assert span_pattern.search(block), (
            "Embed dispatch must open an OTEL span with a name identifying "
            "the embed operation (e.g., 'daemon.dispatch.embed'). Without "
            "this, the GM panel cannot verify embed ran concurrently with "
            "Flux post-37-23. CLAUDE.md OTEL obligation is not satisfied "
            "by the existing 'embed.generated' log line — logs and spans "
            "serve different observers."
        )

    def test_render_dispatch_opens_otel_span(self):
        """Render branch must open an OTEL span naming the dispatch operation.

        Render was previously uninstrumented at the dispatch level; story
        37-23 is the right time to close that gap because the lock split
        introduces the very behavior the span is meant to observe.
        """
        block = _render_handler_block()
        span_pattern = re.compile(
            r"start_as_current_span\s*\(\s*[\"'][^\"']*render[^\"']*[\"']",
            re.IGNORECASE,
        )
        assert span_pattern.search(block), (
            "Render dispatch must open an OTEL span with a name identifying "
            "the render operation (e.g., 'daemon.dispatch.render'). The "
            "dispatch span is what discriminates 'embed ran concurrently "
            "with render' from 'embed waited behind render' in the GM panel."
        )

    def test_embed_span_records_lock_name_attribute(self):
        """The span must carry an attribute identifying the lock held.

        Without a ``lock_name`` attribute, a silent regression to a shared
        lock would still produce dispatch spans with the same names — the
        GM panel could not distinguish the regression. Recording the
        acquired lock is what makes the span a lie detector.
        """
        block = _embed_handler_block()
        # Accept either set_attribute or attributes kwarg on span creation.
        lock_attr_pattern = re.compile(
            r"set_attribute\s*\(\s*[\"']lock_name[\"']\s*,\s*[\"']embed_lock[\"']",
        )
        assert lock_attr_pattern.search(block), (
            "Embed dispatch span must record `lock_name=\"embed_lock\"` as "
            "an attribute. This is the signal the GM panel uses to verify "
            "embed acquired embed_lock (not render_lock) — the 37-23 "
            "invariant made externally observable."
        )

    def test_render_span_records_lock_name_attribute(self):
        block = _render_handler_block()
        lock_attr_pattern = re.compile(
            r"set_attribute\s*\(\s*[\"']lock_name[\"']\s*,\s*[\"']render_lock[\"']",
        )
        assert lock_attr_pattern.search(block), (
            "Render dispatch span must record `lock_name=\"render_lock\"` "
            "as an attribute so the GM panel can observe which lock is "
            "held during which operation."
        )


class TestDocstringDescribesNewInvariant:
    """WorkerPool.embed docstring must accurately describe the post-37-23 contract.

    Positive contract (both required):
      1. Names ``embed_lock`` — tells future readers which lock serializes embed.
      2. Names ``CPU`` — tells future readers embed is off MPS.

    Also a negative: no stale ``render_lock`` reference that implies the old
    shared-lock pattern.

    Additionally: the docstring must not claim ``embed()`` itself acquires the
    lock — the caller holds embed_lock, not the method (see daemon.py dispatch
    in _handle_client). This is a correctness claim about responsibility, not
    just wording style.
    """

    def test_docstring_names_embed_lock(self):
        doc = inspect.getdoc(WorkerPool.embed) or ""
        assert "embed_lock" in doc, (
            "WorkerPool.embed docstring must mention embed_lock so future "
            "readers know which lock serializes embed calls (story 37-23)"
        )

    def test_docstring_names_cpu_device(self):
        doc = inspect.getdoc(WorkerPool.embed) or ""
        assert "CPU" in doc, (
            "WorkerPool.embed docstring must mention CPU — the device "
            "invariant is load-bearing (prevents the 37-5 MPS deadlock "
            "from returning)"
        )

    def test_docstring_has_no_stale_render_lock_reference(self):
        doc = inspect.getdoc(WorkerPool.embed) or ""
        assert "render_lock" not in doc, (
            "WorkerPool.embed docstring still references render_lock — "
            "story 37-23 moved embed off MPS. Update the docstring to "
            "describe embed_lock + CPU device invariant."
        )

    def test_docstring_places_lock_responsibility_on_caller(self):
        """The caller holds embed_lock in _handle_client; pool.embed does not.

        A docstring that implies the method handles serialization (e.g., by
        saying "serialize ... via embed_lock" with the method as the subject)
        misleads future maintainers into thinking they can remove the
        caller-side lock. The docstring MUST positively state that the
        caller is responsible.

        Reject wordings that imply method-as-locker; require wordings that
        make the caller contract explicit.
        """
        doc_lower = (inspect.getdoc(WorkerPool.embed) or "").lower()

        # Negative: method-as-serializer phrasings.
        forbidden_phrases = [
            "acquire embed_lock",
            "acquires embed_lock",
            "acquiring embed_lock",
            "serialize against",
            "serializes against",
            "serializes embed",
        ]
        for phrase in forbidden_phrases:
            assert phrase not in doc_lower, (
                f"WorkerPool.embed docstring contains {phrase!r}, which "
                f"implies the method itself serializes via embed_lock. The "
                f"caller in _handle_client holds the lock. Reword to say "
                f"the caller must hold embed_lock before invoking."
            )

        # Positive: docstring must name the caller's responsibility.
        # Accept any of these phrasings that make caller-ownership explicit.
        caller_phrases = [
            "caller must hold",
            "caller holds",
            "caller-held",
            "caller must acquire",
            "hold embed_lock before",
        ]
        assert any(p in doc_lower for p in caller_phrases), (
            "WorkerPool.embed docstring must positively state that the "
            "CALLER is responsible for holding embed_lock (e.g., 'caller "
            "must hold embed_lock before invoking'). Without this, future "
            "readers won't know the serialization contract lives at the "
            "dispatch site in _handle_client, not in pool.embed itself."
        )
