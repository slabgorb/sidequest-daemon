"""Regression test for story 45-29: span attributes are scoped per-call.

Background: in the Rust client (`sidequest-api`, archived per ADR-082),
``daemon.render`` attached ``art_style`` as a tracing-span field via
``let _guard = span.enter();`` — the guard persisted across ``.await``
points because it was a sync ``enter()`` rather than ``.instrument()``,
so when an ``embed()`` call landed on the same task while a render's
guard was still alive, the embed events inherited the render span's
``art_style`` from the surrounding context. That was the original 37-34
ticket; it was re-scoped to 45-29 in the Rust→Python port-drift audit
(ADR-085) to verify the leak did not survive the port.

The Python port structures every request handler in
``_handle_client`` with its own ``with tracer.start_as_current_span(...)``
block. Each block ends cleanly at scope exit; the current span returns
to whatever was current before, and between requests on the same
connection there is no outer span — each request opens a fresh root
span. The Rust leak cannot manifest by construction.

This file pins that invariant. It drives the real ``_handle_client``
through a render-then-embed sequence and asserts the ``daemon.dispatch.embed``
span carries only embed-related attributes — no ``world``, ``genre``,
``world_style_applied``, ``genre_style_applied``, or any ``render.*``
attribute leaked from the prior render span.

If a future refactor wraps the request loop in an outer span — or any
other change re-introduces parent-context propagation across requests —
this test fails and the Rust-era leak is blocked at review.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sidequest_daemon.media.daemon import _handle_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def otel_exporter(monkeypatch):
    """In-memory OTEL span exporter; resets the global provider per test.

    Mirrors the fixture in ``test_otel_spans.py`` but adds a
    ``ProxyTracer._real_tracer`` reset so that module-level tracers cached
    in production code (``daemon.py``: ``tracer = trace.get_tracer(...)``)
    re-resolve against the new provider. Without that reset, the second
    test in the module run picks up the *previous* test's shut-down provider
    and emits no spans — confirmed by the prior failure mode where test 1
    passed and tests 2-4 saw empty span lists.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        trace,
        "_TRACER_PROVIDER_SET_ONCE",
        trace._TRACER_PROVIDER_SET_ONCE.__class__(),
    )
    trace.set_tracer_provider(provider)

    # Force any module-level ProxyTracers (e.g. daemon.py:86) to re-resolve
    # against the freshly set provider on next span creation. The proxy
    # caches its real tracer on first use; after a provider change we need
    # to invalidate that cache or spans land on the previous provider.
    from sidequest_daemon.media import daemon as daemon_mod

    daemon_mod.tracer = trace.get_tracer("sidequest_daemon.media.daemon")

    yield exporter
    exporter.clear()


# ---------------------------------------------------------------------------
# Test doubles — minimal so the test exercises real handler logic
# ---------------------------------------------------------------------------


class _StubWriter:
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


class _FakePool:
    """Pool stand-in that records calls without touching MPS or sentence-transformers."""

    def __init__(self) -> None:
        self.render_calls = 0
        self.embed_calls = 0
        self.render_started = threading.Event()

    def render(self, params: dict) -> dict:
        self.render_calls += 1
        self.render_started.set()
        time.sleep(0.005)
        return {
            "path": "/tmp/fake.png",
            "tier": params.get("tier"),
            "r2_key": "fake-key",
        }

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [0.0, 0.0, 0.0]


async def _feed_reader(line: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(line)
    reader.feed_eof()
    return reader


def _render_request_line() -> bytes:
    """A render request that bypasses the prompt composer.

    ``positive_prompt`` is supplied directly so ``_handle_client`` skips the
    ``compose_prompt_for`` branch entirely. The render handler still opens
    its full span tree (``daemon.dispatch.render`` + ``render.completed``)
    and the inner spans set the ``world``/``genre``/style attributes that
    this test verifies do not leak.
    """
    return (
        json.dumps(
            {
                "id": "r1",
                "method": "render",
                "params": {
                    "tier": "portrait",
                    "positive_prompt": "a portrait of someone",
                    "world": "aureate_span",
                    "genre": "tea_and_murder",
                    "session_id": "session-1",
                },
            }
        ).encode()
        + b"\n"
    )


def _embed_request_line() -> bytes:
    return (
        json.dumps(
            {
                "id": "e1",
                "method": "embed",
                "params": {"text": "hello world"},
            }
        ).encode()
        + b"\n"
    )


async def _drive_render_then_embed(
    pool: _FakePool,
) -> None:
    """Run render then embed back-to-back on the same `_handle_client` invocation.

    Both frames are fed into the same StreamReader so the request loop in
    ``_handle_client`` processes them sequentially — modeling a single
    daemon connection that handles multiple methods over its lifetime.
    This is the exact production shape the bug would manifest in.
    """
    reader = await _feed_reader(_render_request_line() + _embed_request_line())
    writer = _StubWriter()
    render_lock = asyncio.Lock()
    embed_lock = asyncio.Lock()
    await _handle_client(reader, writer, pool, render_lock, embed_lock)


# ---------------------------------------------------------------------------
# Invariant 1 — embed span carries no style/world/genre/render.* attributes
# ---------------------------------------------------------------------------


# Attribute names that belong to render spans. If any of these surface on
# the embed span, the per-call scoping has been broken.
_RENDER_ONLY_ATTRS = frozenset(
    {
        "world",
        "genre",
        "genre_style_applied",
        "world_style_applied",
        "tier",
        "session_id",
        "r2_key",
        "prompt_length",
        "art_style",  # the original Rust-era attribute name; pinned for documentation
    }
)


_RENDER_NAMESPACE_PREFIX = "render."


@pytest.mark.asyncio
async def test_embed_span_has_no_render_attributes(otel_exporter):
    """Embed span must not carry any attribute that originated on a render span.

    Drives the real production handler with a render request followed by an
    embed request on the same connection. After both requests complete, the
    embed dispatch span is inspected and asserted to contain only
    embed-scoped attributes. Any leak from the prior render span fails the
    test.
    """
    pool = _FakePool()
    await _drive_render_then_embed(pool)

    spans = otel_exporter.get_finished_spans()
    embed_spans = [s for s in spans if s.name == "daemon.dispatch.embed"]
    assert embed_spans, (
        "Expected a 'daemon.dispatch.embed' span from the embed request — "
        f"got: {[s.name for s in spans]}"
    )
    embed_attrs = dict(embed_spans[-1].attributes)

    leaked = {k for k in embed_attrs if k in _RENDER_ONLY_ATTRS}
    assert not leaked, (
        f"Embed span carries render-scoped attribute(s): {sorted(leaked)}. "
        f"Per-request span scoping broken — see story 45-29 for the Rust-era "
        f"leak this prevents."
    )

    leaked_namespace = {
        k for k in embed_attrs if k.startswith(_RENDER_NAMESPACE_PREFIX)
    }
    assert not leaked_namespace, (
        f"Embed span carries 'render.*' namespaced attribute(s): "
        f"{sorted(leaked_namespace)}. The render namespace must not appear "
        f"on embed spans (story 45-29)."
    )


@pytest.mark.asyncio
async def test_embed_span_has_only_expected_attributes(otel_exporter):
    """Positive-shape check: embed span carries the embed-scoped set, nothing more.

    The negative test above catches *known* render attributes leaking.
    This test catches *unknown* attributes leaking — anything that is not
    in the embed allow-list trips the assertion. Together they pin both
    sides of the invariant: no render attrs leak in, no surprise attrs
    appear at all.
    """
    pool = _FakePool()
    await _drive_render_then_embed(pool)

    embed_spans = [
        s
        for s in otel_exporter.get_finished_spans()
        if s.name == "daemon.dispatch.embed"
    ]
    assert embed_spans, "expected daemon.dispatch.embed span"
    attrs = dict(embed_spans[-1].attributes)

    # The embed handler in _handle_client sets exactly these on success.
    expected = {"lock_name", "text_len", "work_ms"}
    unexpected = set(attrs) - expected
    assert not unexpected, (
        f"Embed span carries unexpected attribute(s): {sorted(unexpected)}. "
        f"Allow-list (success path): {sorted(expected)}. Any new attribute "
        f"on the embed span should either be added here or routed through a "
        f"child span — never inherited from a sibling request's span."
    )


# ---------------------------------------------------------------------------
# Invariant 2 — render and embed spans are siblings, not parent/child
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_span_is_root_after_render(otel_exporter):
    """``daemon.dispatch.embed`` must be a root span — no parent context from render.

    A regression where a future refactor wraps the request loop in an outer
    ``with tracer.start_as_current_span(...)`` would silently make every
    embed span a CHILD of that outer span, and (worse) any attributes set
    on the outer span during the render request would parent the embed.
    Pinning ``parent is None`` blocks that pattern at review time.
    """
    pool = _FakePool()
    await _drive_render_then_embed(pool)

    embed_spans = [
        s
        for s in otel_exporter.get_finished_spans()
        if s.name == "daemon.dispatch.embed"
    ]
    assert embed_spans, "expected daemon.dispatch.embed span"
    assert embed_spans[-1].parent is None, (
        "daemon.dispatch.embed has a parent span — request handlers must "
        "not be nested inside an outer span. Each request opens its own "
        "root span (story 45-29)."
    )


# ---------------------------------------------------------------------------
# Invariant 3 — positive control: render's style/world attrs DO appear on render
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_span_does_carry_style_attrs_positive_control(otel_exporter):
    """Sanity check that the test harness actually exercises the render path.

    Without this control, a regression that simply stops emitting the render
    span entirely would let the leak tests pass vacuously (no render span →
    no attributes to leak). This test asserts the render side is live, so
    the negative tests above are meaningful.
    """
    pool = _FakePool()
    await _drive_render_then_embed(pool)

    render_completed = [
        s
        for s in otel_exporter.get_finished_spans()
        if s.name == "render.completed"
    ]
    assert render_completed, (
        "Expected a 'render.completed' span — without it, the test cannot "
        "demonstrate that style attributes are emitted on the render side. "
        "The leak invariant is meaningless without a positive control."
    )
    attrs = dict(render_completed[-1].attributes)
    assert attrs.get("world") == "aureate_span", (
        f"render.completed must record world='aureate_span' to match the "
        f"render request payload, got {attrs.get('world')!r}"
    )
    assert attrs.get("genre") == "tea_and_murder", (
        f"render.completed must record genre='tea_and_murder', got {attrs.get('genre')!r}"
    )
    assert pool.render_calls == 1, "render handler did not invoke pool.render"
    assert pool.embed_calls == 1, "embed handler did not invoke pool.embed"
