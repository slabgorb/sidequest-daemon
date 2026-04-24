import json
from pathlib import Path

from sidequest_daemon.media.preview import main

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "visual_recipes" / "genre_packs"


def _run(argv: list[str], capsys) -> str:
    exit_code = main(argv)
    captured = capsys.readouterr()
    assert exit_code == 0, f"CLI exited {exit_code}: {captured.err}"
    return captured.out


def test_cli_portrait_text_output(capsys, monkeypatch) -> None:
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    out = _run([
        "portrait",
        "--character", "npc:rux",
        "--world", "testworld",
        "--genre", "testgenre",
    ], capsys)
    assert "== Target ==" in out
    assert "== Composed prompt ==" in out
    assert "== Layer breakdown ==" in out
    assert "npc:rux" in out
    assert "inquisitor" in out


def test_cli_illustration_text_output(capsys, monkeypatch) -> None:
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    out = _run([
        "illustration",
        "--participants", "npc:rux,npc:mira",
        "--location", "where:testworld/the_lookout",
        "--action", "arriving at dusk",
        "--camera", "scene",
        "--world", "testworld",
        "--genre", "testgenre",
    ], capsys)
    assert "arriving at dusk" in out
    assert "watchtower" in out


def test_cli_json_output_roundtrips(capsys, monkeypatch) -> None:
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    out = _run([
        "portrait",
        "--character", "npc:rux",
        "--world", "testworld",
        "--genre", "testgenre",
        "--json",
    ], capsys)
    payload = json.loads(out)
    assert payload["worker_type"]
    assert payload["positive_prompt"]
    assert any(layer["slot"] == "CASTING" for layer in payload["layers"])


def test_cli_unknown_character_exits_nonzero(capsys, monkeypatch) -> None:
    monkeypatch.setenv("SIDEQUEST_GENRE_PACKS", str(FIXTURE_ROOT))
    exit_code = main([
        "portrait",
        "--character", "npc:ghost",
        "--world", "testworld",
        "--genre", "testgenre",
    ])
    assert exit_code != 0
    assert "ghost" in capsys.readouterr().err
