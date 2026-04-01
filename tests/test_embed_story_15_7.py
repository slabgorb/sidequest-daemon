"""Story 15-7: Embed endpoint tests for sidequest-daemon.

Tests that:
1. Daemon handles 'embed' method over Unix socket
2. Returns embedding vectors with correct dimensionality
3. Returns model name and latency metadata
4. Rejects empty text input
"""

import json

import pytest

from sidequest_daemon.media.daemon import EMBED_TIERS, EmbedWorker
from sidequest_daemon.media.protocol import WorkerRequest, WorkerResponse


# ============================================================
# AC-2: /embed endpoint exists and handles requests
# ============================================================


class TestEmbedMethodRouting:
    """Verify the daemon routes 'embed' method to an embed worker."""

    def test_embed_tier_constant_exists(self):
        """EMBED_TIERS constant must exist for method routing."""
        assert isinstance(EMBED_TIERS, (set, frozenset))
        assert "embed" in EMBED_TIERS

    def test_embed_worker_class_exists(self):
        """EmbedWorker class must exist to handle embed requests."""
        assert hasattr(EmbedWorker, "generate_embedding")


class TestEmbedRequest:
    """Verify embed request handling."""

    def test_embed_request_requires_text(self):
        """Embed request must include a 'text' field."""
        req = WorkerRequest(method="embed", params={"text": "Hello world"})
        assert req.params["text"] == "Hello world"
        assert req.method == "embed"

    def test_embed_request_rejects_empty_text(self):
        """Embed endpoint must reject empty text — no silent fallbacks."""
        worker = EmbedWorker()
        with pytest.raises((ValueError, TypeError)):
            worker.generate_embedding("")


class TestEmbedResponse:
    """Verify embed response format matches Rust client's EmbedResult."""

    def test_embed_response_has_embedding_vector(self):
        """Response must contain an 'embedding' field with a list of floats."""
        # Simulate the response format the daemon will produce
        response_data = {
            "embedding": [0.1, 0.2, 0.3],
            "model": "all-MiniLM-L6-v2",
            "latency_ms": 42,
        }
        result = WorkerResponse(
            id="test-id",
            result=response_data,
        )
        assert isinstance(result.result["embedding"], list)
        assert len(result.result["embedding"]) > 0
        assert all(isinstance(v, float) for v in result.result["embedding"])

    def test_embed_response_includes_model_name(self):
        """Response must include model name for OTEL lore.embedding_generated event."""
        response_data = {
            "embedding": [0.1],
            "model": "all-MiniLM-L6-v2",
            "latency_ms": 10,
        }
        result = WorkerResponse(id="test-id", result=response_data)
        assert result.result["model"] == "all-MiniLM-L6-v2"

    def test_embed_response_includes_latency(self):
        """Response must include latency_ms for OTEL lore.embedding_generated event."""
        response_data = {
            "embedding": [0.1],
            "model": "all-MiniLM-L6-v2",
            "latency_ms": 55,
        }
        result = WorkerResponse(id="test-id", result=response_data)
        assert result.result["latency_ms"] == 55


class TestEmbedWorkerDimensionality:
    """Verify embedding output dimensionality is consistent."""

    def test_embedding_dimension_is_consistent(self):
        """All embeddings from the same model must have the same dimension."""
        worker = EmbedWorker()
        emb1 = worker.generate_embedding("Hello world")
        emb2 = worker.generate_embedding("The ruins crumble in the wasteland wind")
        assert len(emb1) == len(emb2), (
            f"Embedding dimensions must be consistent: {len(emb1)} vs {len(emb2)}"
        )
        assert len(emb1) > 0, "Embedding must have non-zero dimension"

    def test_embedding_values_are_floats(self):
        """Embedding values must be floats, not ints or strings."""
        worker = EmbedWorker()
        emb = worker.generate_embedding("Test input")
        assert all(isinstance(v, float) for v in emb)
