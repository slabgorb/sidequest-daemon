"""Verify the zimage worker calls r2_writer.upload_artifact and emits r2_key."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _r2_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_S3_ENDPOINT", "https://endpoint.example")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "y")


@pytest.mark.no_r2_stub
def test_worker_emits_r2_key_after_save() -> None:
    from sidequest_daemon.media import r2_writer
    from sidequest_daemon.media.workers import zimage_mlx_worker

    fake_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 64
    captured: dict[str, object] = {}

    def fake_upload(*, world_slug, session_id, kind, content_bytes, content_type):
        captured.update(
            world_slug=world_slug,
            session_id=session_id,
            kind=kind,
            content_type=content_type,
        )
        return f"artifacts/{world_slug}/{session_id}/{kind}/abcd.png"

    with patch.object(r2_writer, "upload_artifact", side_effect=fake_upload):
        result = zimage_mlx_worker.upload_render_to_r2(
            content_bytes=fake_bytes,
            world_slug="dungeon",
            session_id="0d8e",
            kind="portraits",
            content_type="image/png",
        )

    assert result == "artifacts/dungeon/0d8e/portraits/abcd.png"
    assert captured["world_slug"] == "dungeon"
    assert captured["session_id"] == "0d8e"
    assert captured["kind"] == "portraits"
    assert captured["content_type"] == "image/png"
