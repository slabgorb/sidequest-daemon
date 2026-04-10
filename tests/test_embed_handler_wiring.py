"""Wiring tests for the embed handler — guards against the 2026-04-10 deadlock regression.

The original bug had four layers, all in ``sidequest_daemon/media/daemon.py``:

1. ``EmbedWorker()`` was constructed per-request — fresh
   ``SentenceTransformer`` download + MPS placement on every embed call.
2. The handler called ``worker.generate_embedding(text)`` synchronously,
   blocking the asyncio event loop.
3. The handler did not acquire ``render_lock``, so embed could race
   Flux on the same MPS device — the actual deadlock trigger.
4. The startup ``--warmup all`` path only warmed Flux, never the embed
   model — first embed call still paid the cold-load cost mid-gameplay.

These tests assert the structural fixes are present in the source so a
future refactor can't silently regress any layer. Pure source-level
guards (no daemon process required), modeled after
``sidequest-api/.../tests/map_telemetry_wiring_tests.rs``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from sidequest_daemon.media import daemon as daemon_mod
from sidequest_daemon.media.daemon import EmbedWorker, WorkerPool


DAEMON_SOURCE = Path(inspect.getsourcefile(daemon_mod)).read_text()


# ============================================================
# Layer 1.a — Singleton lifecycle on WorkerPool
# ============================================================


class TestWorkerPoolEmbedSingleton:
    """The pool must own the embed worker as a singleton."""

    def test_pool_has_embed_method(self):
        assert hasattr(WorkerPool, "embed"), (
            "WorkerPool.embed missing — embed handler must route through the "
            "singleton, not construct EmbedWorker per request"
        )

    def test_pool_has_warm_up_embed(self):
        assert hasattr(WorkerPool, "warm_up_embed"), (
            "WorkerPool.warm_up_embed missing — embed model must be eagerly "
            "loadable at daemon startup"
        )

    def test_pool_has_ensure_embed(self):
        assert hasattr(WorkerPool, "_ensure_embed")

    def test_pool_init_declares_embed_state(self, tmp_path: Path):
        pool = WorkerPool(tmp_path)
        assert hasattr(pool, "_embed")
        assert hasattr(pool, "_embed_loaded")
        assert pool._embed is None
        assert pool._embed_loaded is False

    def test_pool_status_reports_embed(self, tmp_path: Path):
        pool = WorkerPool(tmp_path)
        status = pool.status()
        assert "embed" in status, (
            "WorkerPool.status() must report embed worker state for the GM panel"
        )
        assert status["embed"] == "cold"

    def test_pool_cleanup_releases_embed(self, tmp_path: Path):
        pool = WorkerPool(tmp_path)
        # Simulate a loaded embed worker
        pool._embed = EmbedWorker()
        pool._embed_loaded = True
        pool.cleanup()
        assert pool._embed is None
        assert pool._embed_loaded is False

    def test_pool_embed_uses_ensure(self):
        """pool.embed must lazily warm before delegating to the worker."""
        src = inspect.getsource(WorkerPool.embed)
        assert "_ensure_embed" in src, (
            "WorkerPool.embed must call _ensure_embed before delegating — "
            "otherwise the singleton can be None on first call"
        )


# ============================================================
# Layer 1.b — Handler routes through singleton + lock + thread
# ============================================================


def _embed_handler_block() -> str:
    """Extract the source of the ``elif method == "embed":`` branch.

    The embed branch is the last ``elif`` in the dispatch, so we bound on
    the ``else:`` that follows it (the UNKNOWN_METHOD fallthrough).
    """
    marker = 'elif method == "embed":'
    start = DAEMON_SOURCE.index(marker)
    end = DAEMON_SOURCE.index("else:", start + len(marker))
    return DAEMON_SOURCE[start:end]


class TestEmbedHandlerWiring:
    """The embed dispatch in _handle_client must be lock-guarded and threaded."""

    def test_handler_routes_through_pool_embed(self):
        block = _embed_handler_block()
        assert "await asyncio.to_thread(pool.embed," in block, (
            "embed handler must route through pool.embed via asyncio.to_thread; "
            "constructing EmbedWorker() inline is the deadlock pattern"
        )

    def test_handler_acquires_render_lock(self):
        block = _embed_handler_block()
        assert "async with render_lock:" in block, (
            "embed handler must acquire render_lock to serialize MPS access "
            "with the Flux render path — otherwise concurrent model sessions "
            "deadlock the Metal driver"
        )

    def test_handler_does_not_construct_worker_inline(self):
        block = _embed_handler_block()
        # Strip comments — the explanatory comment in the handler intentionally
        # mentions ``EmbedWorker()`` to document the regression we're guarding against.
        code_only = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        )
        assert "EmbedWorker()" not in code_only, (
            "embed handler must not construct EmbedWorker() per request — "
            "use the WorkerPool singleton via pool.embed"
        )

    def test_handler_emits_info_log_on_success(self):
        block = _embed_handler_block()
        assert "embed.generated" in block, (
            "embed handler must log embed.generated at INFO so success cases "
            "are visible in daemon logs"
        )

    def test_handler_uses_structured_error(self):
        block = _embed_handler_block()
        assert '"EMBED_FAILED"' in block, (
            "embed failures must surface as structured EMBED_FAILED error — "
            "no silent fallbacks, no zero vectors"
        )


# ============================================================
# Layer 1.c — Startup warmup includes embed
# ============================================================


class TestStartupWarmupIncludesEmbed:
    """``--warmup`` / ``--warmup=all`` must eagerly load the embed model."""

    def test_run_daemon_warms_embed(self):
        run_src = inspect.getsource(daemon_mod._run_daemon)
        assert 'target in ("all", "embed"):' in run_src or "target in ('all', 'embed')" in run_src, (
            "_run_daemon startup warmup must include the embed branch"
        )
        assert "pool.warm_up_embed" in run_src, (
            "_run_daemon must call pool.warm_up_embed during startup warmup"
        )

    def test_warm_up_dispatch_supports_embed(self):
        # The dispatch lives inside _handle_client; check the source.
        marker = 'elif method == "warm_up":'
        start = DAEMON_SOURCE.index(marker)
        end = DAEMON_SOURCE.index("elif method ==", start + len(marker))
        block = DAEMON_SOURCE[start:end]
        assert 'target in ("all", "embed"):' in block or "target in ('all', 'embed')" in block, (
            'warm_up dispatch must support target in ("all", "embed")'
        )
        assert "pool.warm_up_embed" in block


# ============================================================
# Sanity: existing EmbedWorker contract still works (no regression)
# ============================================================


class TestEmbedWorkerContract:
    def test_empty_text_raises(self):
        with pytest.raises((ValueError, TypeError)):
            EmbedWorker().generate_embedding("")
