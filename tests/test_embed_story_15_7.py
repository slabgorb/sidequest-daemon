"""Story 15-7: Embed endpoint tests for sidequest-daemon.

Tests that:
1. Daemon has an EMBED_TIERS constant so method routing sees "embed"
2. EmbedWorker exposes generate_embedding
3. EmbedWorker rejects empty text (no silent fallback)
4. Embedding dimension and value type are consistent
"""

import pytest

from sidequest_daemon.media.daemon import EMBED_TIERS, EmbedWorker


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

    def test_embed_request_rejects_empty_text(self):
        """Embed endpoint must reject empty text — no silent fallbacks."""
        worker = EmbedWorker()
        with pytest.raises((ValueError, TypeError)):
            worker.generate_embedding("")


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
