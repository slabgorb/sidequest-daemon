"""Unit tests for r2_writer.upload_artifact (mocked S3)."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from sidequest_daemon.media import r2_writer


def _bytes(payload: bytes = b"x") -> bytes:
    return payload * 32


def _expected_key(world: str, session: str, kind: str, content: bytes, ext: str) -> str:
    sha = hashlib.sha256(content).hexdigest()
    return f"artifacts/{world}/{session}/{kind}/{sha}.{ext}"


@pytest.fixture
def fake_client() -> MagicMock:
    return MagicMock()


def test_upload_artifact_returns_relative_path(fake_client: MagicMock) -> None:
    content = _bytes(b"abc")
    expected = _expected_key("w1", "s1", "portraits", content, "png")
    with patch.object(r2_writer, "_client", lambda: fake_client):
        rel = r2_writer.upload_artifact(
            world_slug="w1",
            session_id="s1",
            kind="portraits",
            content_bytes=content,
            content_type="image/png",
        )
    assert rel == expected
    fake_client.put_object.assert_called_once()
    kwargs = fake_client.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "sidequest"
    assert kwargs["Key"] == expected
    assert kwargs["ContentType"] == "image/png"
    assert kwargs["CacheControl"] == "public, max-age=86400"
    assert kwargs["Body"] == content


def test_upload_artifact_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        r2_writer.upload_artifact(
            world_slug="w",
            session_id="s",
            kind="bogus",  # type: ignore[arg-type]
            content_bytes=b"x",
            content_type="image/png",
        )


def test_upload_artifact_unknown_content_type_raises() -> None:
    with pytest.raises(ValueError, match="content_type"):
        r2_writer.upload_artifact(
            world_slug="w",
            session_id="s",
            kind="portraits",
            content_bytes=b"x",
            content_type="application/x-bogus",
        )


def test_upload_artifact_propagates_client_errors(fake_client: MagicMock) -> None:
    fake_client.put_object.side_effect = RuntimeError("boom")
    with patch.object(r2_writer, "_client", lambda: fake_client):
        with pytest.raises(RuntimeError, match="boom"):
            r2_writer.upload_artifact(
                world_slug="w",
                session_id="s",
                kind="portraits",
                content_bytes=b"x",
                content_type="image/png",
            )


@pytest.mark.parametrize(
    "kind,ctype,ext",
    [
        ("portraits", "image/png", "png"),
        ("poi", "image/png", "png"),
        ("scenes", "image/jpeg", "jpg"),
        ("music", "audio/ogg", "ogg"),
        ("sfx", "audio/ogg", "ogg"),
    ],
)
def test_upload_artifact_extension_for_content_type(
    fake_client: MagicMock, kind: str, ctype: str, ext: str
) -> None:
    content = b"data" * 16
    expected = _expected_key("w", "s", kind, content, ext)
    with patch.object(r2_writer, "_client", lambda: fake_client):
        rel = r2_writer.upload_artifact(
            world_slug="w",
            session_id="s",
            kind=kind,  # type: ignore[arg-type]
            content_bytes=content,
            content_type=ctype,
        )
    assert rel == expected
