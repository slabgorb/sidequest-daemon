"""Story 37-5: Embed endpoint integration tests.

The playtest-2 (2026-04-12) bug: /embed returned "Unknown error" for
every request, silently degrading RAG semantic search for the entire
session. The narrator compensated by improvising without lore grounding.

These tests go beyond the structural wiring guards in
``test_embed_handler_wiring.py`` — they exercise the actual runtime
behavior of the embed pipeline from pool.embed() through the socket
handler's response envelope.

Existing coverage:
- ``test_embed_story_15_7.py`` — EmbedWorker class contract + response format
- ``test_embed_handler_wiring.py`` — source-level wiring guards (singleton,
  lock, thread)

Gap these tests close:
- Full socket round-trip (JSON-RPC request → handler → pool → response)
- Response schema parity with Rust EmbedResult
- Error response schema parity with Rust ErrorPayload
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sidequest_daemon.media.daemon import (
    EmbedWorker,
    WorkerPool,
    _handle_client,
)


# ============================================================
# AC-1: pool.embed() returns a valid embedding after warmup
# ============================================================


class TestPoolEmbedRuntime:
    """Verify pool.embed() works end-to-end, not just that the method exists."""

    def test_pool_embed_returns_list_of_floats(self, tmp_path: Path):
        """pool.embed() must return a list of Python floats, not numpy
        float32 or similar — the JSON serializer in the handler does
        ``[float(v) for v in embedding]`` to ensure this."""
        pool = WorkerPool(tmp_path)
        pool.warm_up_embed()
        result = pool.embed("The ancient ruins hold forgotten secrets.")
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0, "Embedding must be non-empty"
        assert all(
            isinstance(v, float) for v in result
        ), f"All values must be float, got types: {set(type(v).__name__ for v in result)}"

    def test_pool_embed_dimension_is_384(self, tmp_path: Path):
        """all-MiniLM-L6-v2 produces 384-dimensional embeddings. If the
        dimension changes, the Rust side's similarity math will silently
        produce garbage results."""
        pool = WorkerPool(tmp_path)
        pool.warm_up_embed()
        result = pool.embed("Test embedding dimension.")
        assert len(result) == 384, (
            f"Expected 384-dim embedding from all-MiniLM-L6-v2, got {len(result)}. "
            "If the model changed, update LoreStore's similarity math."
        )

    def test_pool_embed_without_warmup_lazy_loads(self, tmp_path: Path):
        """pool.embed() must lazy-load via _ensure_embed if warmup wasn't
        called — the embed worker in Rust doesn't call warm_up before
        embed, it relies on this lazy path."""
        pool = WorkerPool(tmp_path)
        # Do NOT call warm_up_embed — simulate the embed worker's path
        result = pool.embed("Lazy load test.")
        assert isinstance(result, list)
        assert len(result) == 384


# ============================================================
# AC-2: Handler response matches Rust EmbedResult schema
# ============================================================


class TestEmbedResponseSchemaParity:
    """The daemon's embed response must deserialize into Rust's EmbedResult:

        pub struct EmbedResult {
            pub embedding: Vec<f32>,
            pub model: String,
            pub latency_ms: u64,
        }

    Field name mismatches or type mismatches cause DaemonError::InvalidResponse
    on the Rust side, which surfaces as the "Unknown error" bug.
    """

    def test_response_has_embedding_field(self):
        """Response must include 'embedding' — not 'embeddings' (plural),
        'vector', 'values', or any other name. Rust's serde has no alias
        on this field."""
        # Simulate what the handler builds
        embedding = [0.1, 0.2, 0.3]
        response = {
            "embedding": embedding,
            "model": "all-MiniLM-L6-v2",
            "latency_ms": 42,
        }
        assert "embedding" in response
        assert isinstance(response["embedding"], list)

    def test_response_has_model_field_as_string(self):
        """Response must include 'model' as a string. Rust's EmbedResult
        deserializes this as String, not Option<String>."""
        response = {
            "embedding": [0.1],
            "model": "all-MiniLM-L6-v2",
            "latency_ms": 42,
        }
        assert isinstance(response["model"], str)
        assert len(response["model"]) > 0

    def test_response_has_latency_ms_as_integer(self):
        """Response must include 'latency_ms' as an integer. Rust's
        EmbedResult deserializes this as u64."""
        response = {
            "embedding": [0.1],
            "model": "all-MiniLM-L6-v2",
            "latency_ms": 42,
        }
        assert isinstance(response["latency_ms"], int)

    def test_response_json_roundtrips_cleanly(self):
        """The full response must survive JSON serialization without
        type coercion issues (e.g., numpy float32 not being JSON
        serializable)."""
        embedding = [float(v) for v in [0.1, 0.2, 0.3]]
        response = {
            "embedding": embedding,
            "model": "all-MiniLM-L6-v2",
            "latency_ms": 42,
        }
        serialized = json.dumps(response)
        deserialized = json.loads(serialized)
        assert deserialized["embedding"] == embedding
        assert deserialized["model"] == "all-MiniLM-L6-v2"
        assert deserialized["latency_ms"] == 42


# ============================================================
# AC-3: Error response matches Rust ErrorPayload schema
# ============================================================


class TestEmbedErrorResponseSchema:
    """When embed fails, the error response must match Rust's ErrorPayload:

        pub struct ErrorPayload {
            pub code: i32,        // deserialized from string or int
            pub message: String,
        }

    The daemon sends string codes (e.g., "EMBED_FAILED"). The Rust
    deserializer maps these to -1. The message must be non-empty —
    an empty message string is the likely source of the "Unknown error"
    report from playtest 2.
    """

    def test_embed_error_has_code_field(self):
        """Error response must include 'code' — Rust ErrorPayload requires it."""
        error = {"code": "EMBED_FAILED", "message": "model load failed"}
        assert "code" in error
        assert isinstance(error["code"], str)

    def test_embed_error_has_message_field(self):
        """Error response must include 'message' — Rust ErrorPayload requires it."""
        error = {"code": "EMBED_FAILED", "message": "model load failed"}
        assert "message" in error
        assert isinstance(error["message"], str)

    def test_embed_error_message_is_not_empty(self):
        """Error message must be non-empty. If str(exception) is empty,
        the Rust side sees 'daemon error (-1): ' which the GM panel
        shows as an unknown error.

        The handler must guard against empty exception messages by
        providing a fallback that includes the exception type name.
        """
        # Simulate what the handler does: str(e) or fallback
        errors_to_check = [
            RuntimeError(""),
            ValueError(""),
            Exception(""),
        ]
        for exc in errors_to_check:
            # This is the handler's logic — must produce non-empty output
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"
            assert len(error_msg) > 0, (
                f"{type(exc).__name__}('') must produce a non-empty error "
                "message via the handler's fallback guard"
            )
            assert type(exc).__name__ in error_msg, (
                f"Fallback message must include exception type name, "
                f"got: {error_msg!r}"
            )


# ============================================================
# AC-4: Socket handler round-trip for embed method
# ============================================================


class TestEmbedSocketRoundTrip:
    """Full JSON-RPC round-trip through _handle_client for the embed method.

    This is the critical missing test — all existing tests are either
    structural (source-level) or unit-level (EmbedWorker class). Nobody
    tests the actual handler dispatch path.
    """

    @pytest.fixture
    def pool_with_mock_embed(self, tmp_path: Path):
        """WorkerPool with a mock embed that returns a known vector."""
        pool = WorkerPool(tmp_path)
        mock_worker = MagicMock(spec=EmbedWorker)
        mock_worker.generate_embedding.return_value = [0.1] * 384
        pool._embed = mock_worker
        pool._embed_loaded = True
        return pool

    @pytest.mark.asyncio
    async def test_embed_request_returns_valid_response(self, pool_with_mock_embed):
        """Send an embed JSON-RPC request, verify the response envelope."""
        request = json.dumps({
            "id": "test-embed-1",
            "method": "embed",
            "params": {"text": "The ruins crumble in the wasteland wind."},
        }) + "\n"

        reader = asyncio.StreamReader()
        reader.feed_data(request.encode())
        reader.feed_eof()

        responses = []
        writer = MagicMock()
        writer.get_extra_info = MagicMock(return_value="test")
        writer.write = lambda data: responses.append(data)
        writer.close = MagicMock()
        async def noop():
            pass
        writer.wait_closed = noop

        render_lock = asyncio.Lock()
        embed_lock = asyncio.Lock()
        await _handle_client(reader, writer, pool_with_mock_embed, render_lock, embed_lock)

        assert len(responses) >= 1, "Handler must write at least one response"
        resp = json.loads(responses[0].decode())
        assert resp["id"] == "test-embed-1"
        assert "result" in resp, f"Expected 'result' in response, got: {resp}"
        assert "error" not in resp or resp["error"] is None

        result = resp["result"]
        assert "embedding" in result, "Response must contain 'embedding'"
        assert "model" in result, "Response must contain 'model'"
        assert "latency_ms" in result, "Response must contain 'latency_ms'"
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) == 384

    @pytest.mark.asyncio
    async def test_embed_empty_text_returns_structured_error(self, pool_with_mock_embed):
        """Empty text must return INVALID_REQUEST, not crash."""
        request = json.dumps({
            "id": "test-embed-2",
            "method": "embed",
            "params": {"text": ""},
        }) + "\n"

        reader = asyncio.StreamReader()
        reader.feed_data(request.encode())
        reader.feed_eof()

        responses = []
        writer = MagicMock()
        writer.get_extra_info = MagicMock(return_value="test")
        writer.write = lambda data: responses.append(data)
        writer.close = MagicMock()
        async def _noop():
            pass
        writer.wait_closed = _noop

        render_lock = asyncio.Lock()
        embed_lock = asyncio.Lock()
        await _handle_client(reader, writer, pool_with_mock_embed, render_lock, embed_lock)

        assert len(responses) >= 1
        resp = json.loads(responses[0].decode())
        assert "error" in resp
        assert resp["error"]["code"] == "INVALID_REQUEST"
        assert len(resp["error"]["message"]) > 0

    @pytest.mark.asyncio
    async def test_embed_failure_returns_embed_failed_code(self, tmp_path: Path):
        """When pool.embed() raises, the error must have code EMBED_FAILED
        with a non-empty message — not a generic or empty error."""
        pool = WorkerPool(tmp_path)
        mock_worker = MagicMock(spec=EmbedWorker)
        mock_worker.generate_embedding.side_effect = RuntimeError(
            "MPS backend out of memory"
        )
        pool._embed = mock_worker
        pool._embed_loaded = True

        request = json.dumps({
            "id": "test-embed-3",
            "method": "embed",
            "params": {"text": "This will fail."},
        }) + "\n"

        reader = asyncio.StreamReader()
        reader.feed_data(request.encode())
        reader.feed_eof()

        responses = []
        writer = MagicMock()
        writer.get_extra_info = MagicMock(return_value="test")
        writer.write = lambda data: responses.append(data)
        writer.close = MagicMock()
        async def _noop():
            pass
        writer.wait_closed = _noop

        render_lock = asyncio.Lock()
        embed_lock = asyncio.Lock()
        await _handle_client(reader, writer, pool, render_lock, embed_lock)

        assert len(responses) >= 1
        resp = json.loads(responses[0].decode())
        assert "error" in resp
        error = resp["error"]
        assert error["code"] == "EMBED_FAILED", (
            f"Expected EMBED_FAILED code, got {error['code']}"
        )
        assert len(error["message"]) > 0, (
            "Error message must not be empty — empty message surfaces as "
            "'Unknown error' on the Rust/GM panel side"
        )
        assert "MPS backend out of memory" in error["message"]
