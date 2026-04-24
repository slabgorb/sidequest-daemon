"""Wiring test (per CLAUDE.md): prove daemon config loads cleanly at boot."""

from pathlib import Path

import pytest


def test_daemon_refuses_to_start_with_invalid_recipes(tmp_path) -> None:
    from sidequest_daemon.media import daemon as daemon_module

    bad = tmp_path / "recipes.yaml"
    bad.write_text("portrait: {kind: portrait, direction_camera: fabricated_shot}")
    with pytest.raises(ValueError):
        daemon_module.validate_startup_config(
            recipes_path=bad,
            cameras_path=Path(__file__).resolve().parents[1] / "cameras.yaml",
        )


def test_daemon_accepts_valid_config() -> None:
    from sidequest_daemon.media import daemon as daemon_module

    root = Path(__file__).resolve().parents[1]
    daemon_module.validate_startup_config(
        recipes_path=root / "recipes.yaml",
        cameras_path=root / "cameras.yaml",
    )
