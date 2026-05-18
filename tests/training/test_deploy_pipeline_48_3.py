"""Failing tests for story 48-3: close the MLX -> Ollama serving gap.

RED phase (TEA / Radar O'Reilly). These tests fail until Dev creates the
net-new CI-safe layer that turns a trained MLX adapter into a served,
genre-tagged Ollama model:

  - ``sidequest_daemon/training/corpus_gate.py``
      ``CORPUS_GATE_MIN_PAIRS`` (int, == 500), ``CorpusGateError``,
      ``CorpusGateResult`` (passed/total/threshold/reason),
      ``evaluate_corpus_gate(total, *, min_pairs=...) -> CorpusGateResult``,
      ``enforce_corpus_gate(total, *, min_pairs=...) -> CorpusGateResult``
      (raises ``CorpusGateError`` loudly when not passed).
  - ``sidequest_daemon/training/gguf_convert.py``
      ``GgufConversionError``,
      ``convert_lora_to_gguf(adapter_dir, out_path, *, runner=None) -> Path``
  - ``sidequest_daemon/training/ollama_modelfile.py``
      ``OllamaModelfileError``, ``model_tag(genre) -> str``,
      ``render_modelfile(base_model, adapter_gguf) -> str``,
      ``create_ollama_model(genre, base_model, adapter_gguf, *,
      modelfile_dir, runner=None) -> str``
  - ``sidequest_daemon/training/deploy_cli.py``
      module-level ``EXIT_PASS`` (0), ``EXIT_CONFIG_ERROR`` (!=0),
      ``EXIT_TOOLING_UNAVAILABLE`` (!=0), ``OPERATOR_NOTE`` (str),
      ``main(argv) -> int`` -- the non-test consumer that wires
      gate -> gguf -> modelfile into one pipeline.

Two-layer split (mirrors 48-2 / 48-4): every test here is CI-safe. The
subprocess boundary (llama.cpp convert, ``ollama create``) is dependency-
injected exactly like ``trainer.py``'s ``trainer_fn``; the live training run
+ live ``ollama create`` are operator-evidence only and run on Keith's M3
Ultra. No test in this file shells out.

Authoritative spec source: ``.session/48-3-session.md`` Technical Context,
substeps (a) corpus gate, (c) GGUF conversion, (d) Ollama Modelfile, plus
the SM Assessment's "must be surfaced loudly, not silently defaulted" gate
requirement.

Rule-enforcement (.pennyfarthing/gates/lang-review/python.md):
  #1  silent-exceptions      -- test_enforce_corpus_gate_raises_loudly,
                               test_gguf_convert_nonzero_return_raises,
                               test_gguf_convert_missing_adapter_fails_loudly
  #2  mutable-defaults       -- test_rule2_no_mutable_default_args
  #3  type-annotations       -- test_rule3_public_api_fully_annotated
  #8  unsafe-deserialization -- test_rule8_no_shell_true_in_subprocess,
                               test_gguf_convert_builds_list_argv_not_shell
  #5  path-handling          -- test_create_ollama_model_writes_utf8
  #11 input-validation       -- test_model_tag_blank_genre_rejected,
                               test_deploy_cli_missing_corpus_is_config_error
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from sidequest_daemon.training.corpus_gate import (
    CORPUS_GATE_MIN_PAIRS,
    CorpusGateError,
    enforce_corpus_gate,
    evaluate_corpus_gate,
)
from sidequest_daemon.training.gguf_convert import (
    GgufConversionError,
    convert_lora_to_gguf,
)
from sidequest_daemon.training.ollama_modelfile import (
    OllamaModelfileError,
    create_ollama_model,
    model_tag,
    render_modelfile,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mined_sample.jsonl"
DAEMON_PKG = Path(__file__).resolve().parents[2] / "sidequest_daemon"


class _FakeProc:
    """subprocess.CompletedProcess-compatible stand-in (returncode only)."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr


def _load_deploy_cli() -> object:
    """Import deploy_cli as a module object.

    A missing module raises ImportError -- the intended RED signal, not a
    skip (CLAUDE.md: No Silent Fallbacks).
    """
    import sidequest_daemon.training.deploy_cli as mod

    return mod


# ----------------------------------------------------------------------- #
# Substep (a) -- corpus gate. The existing ``_warn_low_volume`` in cli.py is
# a SOFT warning that proceeds anyway; substep (a) demands a HARD gate that
# refuses to ship an overfit adapter and surfaces the decision loudly.
# ----------------------------------------------------------------------- #


def test_corpus_gate_threshold_is_500() -> None:
    assert CORPUS_GATE_MIN_PAIRS == 500


def test_corpus_gate_passes_at_threshold_boundary() -> None:
    result = evaluate_corpus_gate(500)
    assert result.passed is True
    assert result.total == 500
    assert result.threshold == 500


def test_corpus_gate_fails_just_below_threshold() -> None:
    result = evaluate_corpus_gate(499)
    assert result.passed is False
    assert result.total == 499
    assert "499" in result.reason
    assert "500" in result.reason


def test_corpus_gate_empty_corpus_fails() -> None:
    result = evaluate_corpus_gate(0)
    assert result.passed is False


def test_enforce_corpus_gate_raises_loudly_when_insufficient() -> None:
    """Rule #1 + project 'No Silent Fallbacks': an insufficient corpus must
    NOT silently default to training an overfit adapter -- it must raise.
    """
    with pytest.raises(CorpusGateError) as exc:
        enforce_corpus_gate(10)
    assert "10" in str(exc.value)
    assert "500" in str(exc.value)


def test_enforce_corpus_gate_returns_result_when_sufficient() -> None:
    result = enforce_corpus_gate(1000)
    assert result.passed is True
    assert result.total == 1000


# ----------------------------------------------------------------------- #
# Substep (c) -- MLX safetensors -> GGUF conversion (subprocess boundary
# dependency-injected; the real llama.cpp run is operator-evidence only).
# ----------------------------------------------------------------------- #


def test_gguf_convert_missing_adapter_fails_loudly(tmp_path: Path) -> None:
    """No adapters.safetensors present -> fail BEFORE invoking any tool.

    The injected runner must never be called: a missing input is a config
    error surfaced loudly, not a silent no-op.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    out = tmp_path / "adapter.gguf"
    called: list[list[str]] = []

    def runner(argv: list[str]) -> _FakeProc:
        called.append(argv)
        return _FakeProc(0)

    with pytest.raises(GgufConversionError):
        convert_lora_to_gguf(adapter_dir, out, runner=runner)
    assert called == []


def test_gguf_convert_builds_list_argv_not_shell(tmp_path: Path) -> None:
    """Rule #8: the converter must be invoked with a list argv (no
    shell=True / string interpolation -> command injection, CWE-78).
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 8)
    out = tmp_path / "adapter.gguf"
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> _FakeProc:
        seen.append(argv)
        out.write_bytes(b"GGUF\x00")
        return _FakeProc(0)

    convert_lora_to_gguf(adapter_dir, out, runner=runner)
    assert len(seen) == 1
    argv = seen[0]
    assert isinstance(argv, list)
    assert all(isinstance(tok, str) for tok in argv)
    joined = " ".join(argv)
    assert "adapters.safetensors" in joined
    assert str(out) in joined


def test_gguf_convert_nonzero_return_raises(tmp_path: Path) -> None:
    """Rule #1: a failed converter exit must raise, not be swallowed."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 8)
    out = tmp_path / "adapter.gguf"

    def runner(argv: list[str]) -> _FakeProc:
        return _FakeProc(returncode=1, stderr="convert exploded")

    with pytest.raises(GgufConversionError):
        convert_lora_to_gguf(adapter_dir, out, runner=runner)


def test_gguf_convert_success_returns_out_path(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"\x00" * 8)
    out = tmp_path / "adapter.gguf"

    def runner(argv: list[str]) -> _FakeProc:
        out.write_bytes(b"GGUF\x00")
        return _FakeProc(0)

    result = convert_lora_to_gguf(adapter_dir, out, runner=runner)
    assert result == out
    assert out.exists()


# ----------------------------------------------------------------------- #
# Substep (d) -- Ollama Modelfile generation + ``ollama create``.
# ----------------------------------------------------------------------- #


def test_model_tag_format() -> None:
    assert (
        model_tag("caverns_and_claudes")
        == "sidequest-narrator-caverns_and_claudes:latest"
    )


@pytest.mark.parametrize("bad", ["", "   "])
def test_model_tag_blank_genre_rejected(bad: str) -> None:
    """Rule #11: a blank genre at this boundary is operator nonsense."""
    with pytest.raises(ValueError):
        model_tag(bad)


def test_render_modelfile_has_from_and_adapter(tmp_path: Path) -> None:
    gguf = tmp_path / "adapter.gguf"
    text = render_modelfile("qwen2.5:7b-instruct", gguf)
    assert "FROM qwen2.5:7b-instruct" in text
    assert f"ADAPTER {gguf}" in text


def test_render_modelfile_blank_base_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_modelfile("", tmp_path / "adapter.gguf")


def test_create_ollama_model_writes_utf8_and_invokes_create(
    tmp_path: Path,
) -> None:
    """Rule #5: the Modelfile is written with an explicit encoding and
    ``ollama create`` is invoked with a list argv (rule #8)."""
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(b"GGUF\x00")
    mfdir = tmp_path / "mf"
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> _FakeProc:
        seen.append(argv)
        return _FakeProc(0)

    tag = create_ollama_model(
        "caverns_and_claudes",
        "qwen2.5:7b-instruct",
        gguf,
        modelfile_dir=mfdir,
        runner=runner,
    )
    assert tag == "sidequest-narrator-caverns_and_claudes:latest"

    written = list(mfdir.rglob("Modelfile"))
    assert len(written) == 1
    body = written[0].read_text(encoding="utf-8")
    assert "FROM qwen2.5:7b-instruct" in body

    assert len(seen) == 1
    argv = seen[0]
    assert isinstance(argv, list)
    assert argv[:3] == ["ollama", "create", tag]
    assert "-f" in argv


def test_create_ollama_model_nonzero_raises(tmp_path: Path) -> None:
    """Rule #1: a failed ``ollama create`` must raise loudly."""
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(b"GGUF\x00")

    def runner(argv: list[str]) -> _FakeProc:
        return _FakeProc(returncode=1, stderr="ollama: no such base model")

    with pytest.raises(OllamaModelfileError):
        create_ollama_model(
            "caverns_and_claudes",
            "qwen2.5:7b-instruct",
            gguf,
            modelfile_dir=tmp_path / "mf",
            runner=runner,
        )


# ----------------------------------------------------------------------- #
# Deploy pipeline CLI -- exit-code taxonomy + operator-evidence no-op +
# wiring. Mirrors 48-4 ab_eval_harness_cli.py / 48-2 ollama_latency_check.py.
# ----------------------------------------------------------------------- #


def test_deploy_cli_defines_exit_constants() -> None:
    cli = _load_deploy_cli()
    for const in ("EXIT_PASS", "EXIT_CONFIG_ERROR", "EXIT_TOOLING_UNAVAILABLE"):
        assert hasattr(cli, const), f"deploy_cli must define {const}"
        assert isinstance(getattr(cli, const), int)
    assert cli.EXIT_PASS == 0
    assert cli.EXIT_CONFIG_ERROR != 0
    assert cli.EXIT_TOOLING_UNAVAILABLE != 0
    distinct = {
        cli.EXIT_PASS,
        cli.EXIT_CONFIG_ERROR,
        cli.EXIT_TOOLING_UNAVAILABLE,
    }
    assert len(distinct) == 3


def test_deploy_cli_below_gate_is_config_error(tmp_path: Path) -> None:
    """Substep (a) wired: a real (non-smoke) corpus below the 500-pair gate
    must refuse with EXIT_CONFIG_ERROR, not silently train an overfit
    adapter. The 3-pair fixture is intentionally far below the gate.
    """
    cli = _load_deploy_cli()
    rc = cli.main(
        [
            "--corpus",
            str(FIXTURE),
            "--base",
            "qwen2.5:7b-instruct",
            "--genre",
            "caverns_and_claudes",
            "--out",
            str(tmp_path / "loras"),
            "--output-md",
            str(tmp_path / "deploy.md"),
        ]
    )
    assert rc == cli.EXIT_CONFIG_ERROR


def test_deploy_cli_missing_corpus_is_config_error(tmp_path: Path) -> None:
    """Rule #11: a nonexistent --corpus path is a config error."""
    cli = _load_deploy_cli()
    rc = cli.main(
        [
            "--corpus",
            str(tmp_path / "nope.jsonl"),
            "--base",
            "qwen2.5:7b-instruct",
            "--genre",
            "caverns_and_claudes",
            "--out",
            str(tmp_path / "loras"),
            "--output-md",
            str(tmp_path / "deploy.md"),
        ]
    )
    assert rc == cli.EXIT_CONFIG_ERROR


def test_deploy_cli_tooling_absent_writes_operator_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator-evidence no-op (mirrors 48-4 AC4): when the conversion
    tooling is absent (CI machine, no llama.cpp / ollama), the CLI exits
    with the distinct tooling-unavailable code and writes a note pointing
    the operator at the M3 Ultra -- it does NOT crash with a traceback.

    Drives the real CLI; only the tooling boundary is substituted. This
    deliberately avoids the 48-4 anti-pattern of monkeypatching the whole
    unit so the production flow stays untested.
    """
    cli = _load_deploy_cli()
    out_md = tmp_path / "deploy.md"

    def _explode(*a: object, **k: object) -> object:
        raise FileNotFoundError("llama.cpp convert_lora_to_gguf not found")

    monkeypatch.setattr(cli, "convert_lora_to_gguf", _explode)

    big = tmp_path / "big.jsonl"
    row = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    big.write_text((row + "\n") * 600, encoding="utf-8")

    rc = cli.main(
        [
            "--corpus",
            str(big),
            "--base",
            "qwen2.5:7b-instruct",
            "--genre",
            "caverns_and_claudes",
            "--out",
            str(tmp_path / "loras"),
            "--output-md",
            str(out_md),
            "--smoke",
        ]
    )
    assert rc == cli.EXIT_TOOLING_UNAVAILABLE
    note = out_md.read_text(encoding="utf-8").lower() if out_md.exists() else ""
    assert "operator" in note or "m3" in note or "ollama" in note


# ----------------------------------------------------------------------- #
# Rule-enforcement (signature / source scans).
# ----------------------------------------------------------------------- #


def test_rule2_no_mutable_default_args() -> None:
    for fn in (
        evaluate_corpus_gate,
        enforce_corpus_gate,
        convert_lora_to_gguf,
        render_modelfile,
        create_ollama_model,
    ):
        sig = inspect.signature(fn)
        for name, param in sig.parameters.items():
            assert not isinstance(param.default, (list, dict, set)), (
                f"{fn.__qualname__} param {name!r} has mutable default"
            )


def test_rule3_public_api_fully_annotated() -> None:
    for fn in (
        evaluate_corpus_gate,
        enforce_corpus_gate,
        convert_lora_to_gguf,
        model_tag,
        render_modelfile,
        create_ollama_model,
    ):
        sig = inspect.signature(fn)
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"{fn.__qualname__} missing return annotation"
        )
        for name, param in sig.parameters.items():
            assert param.annotation is not inspect.Parameter.empty, (
                f"{fn.__qualname__} param {name!r} missing annotation"
            )


def test_rule8_no_shell_true_in_subprocess() -> None:
    """Rule #8: none of the net-new daemon modules may invoke a subprocess
    with shell=True or os.system (command injection, CWE-78).
    """
    targets = [
        DAEMON_PKG / "training" / "gguf_convert.py",
        DAEMON_PKG / "training" / "ollama_modelfile.py",
        DAEMON_PKG / "training" / "deploy_cli.py",
    ]
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "system"
                ):
                    raise AssertionError(f"{path.name}: os.system used")
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(
                        kw.value, ast.Constant
                    ):
                        assert kw.value.value is not True, (
                            f"{path.name}: subprocess shell=True"
                        )


# ----------------------------------------------------------------------- #
# Wiring (CLAUDE.md: no half-wired features) -- the deploy CLI must consume
# all three pipeline components so substeps (a)/(c)/(d) are connected, not
# three orphan functions.
# ----------------------------------------------------------------------- #


def test_wiring_deploy_cli_consumes_pipeline_components() -> None:
    cli_path = DAEMON_PKG / "training" / "deploy_cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
    for symbol in (
        "enforce_corpus_gate",
        "convert_lora_to_gguf",
        "create_ollama_model",
    ):
        assert symbol in imported, (
            f"deploy_cli must import {symbol} -- a pipeline component with "
            f"no non-test consumer is not wired"
        )


# ----------------------------------------------------------------------- #
# CI-safe self-proof (mirrors 48-4 AC3): this suite must not shell out.
# ----------------------------------------------------------------------- #


def test_ci_safe_no_real_subprocess_in_suite() -> None:
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(
            node.func, ast.Attribute
        ):
            attr = node.func.attr
            assert attr not in {"run", "Popen", "call", "system"}, (
                "CI-safe violation: suite invokes a real subprocess; use "
                "the injected runner / monkeypatch instead"
            )
