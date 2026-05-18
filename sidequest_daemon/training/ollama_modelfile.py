"""Ollama Modelfile generation + ``ollama create`` for story 48-3 substep (d).

Writes a Modelfile whose ``ADAPTER`` directive points at the GGUF-converted
LoRA, then registers it as ``sidequest-narrator-<genre>:latest``. The
``ollama create`` invocation is dependency-injected so CI can assert argv
and failure handling without a live Ollama (operator-evidence, M3 Ultra).
"""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class OllamaModelfileError(RuntimeError):
    """Raised when ``ollama create`` exits non-zero."""


def _require(value: str, field: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be a non-empty string")
    return stripped


def model_tag(genre: str) -> str:
    """Return the Ollama model tag for a genre's specialized narrator."""
    return f"sidequest-narrator-{_require(genre, 'genre')}:latest"


def render_modelfile(base_model: str, adapter_gguf: Path) -> str:
    """Render Modelfile text wiring ``base_model`` + the GGUF LoRA adapter."""
    base = _require(base_model, "base_model")
    return f"FROM {base}\nADAPTER {Path(adapter_gguf)}\n"


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True)  # noqa: S603


def create_ollama_model(
    genre: str,
    base_model: str,
    adapter_gguf: Path,
    *,
    modelfile_dir: Path,
    runner: Runner | None = None,
) -> str:
    """Write the Modelfile and ``ollama create`` the genre narrator model.

    Returns the created model tag. Raises :class:`OllamaModelfileError`
    if ``ollama create`` exits non-zero.
    """
    tag = model_tag(genre)
    body = render_modelfile(base_model, adapter_gguf)

    mfdir = Path(modelfile_dir)
    mfdir.mkdir(parents=True, exist_ok=True)
    modelfile_path = mfdir / "Modelfile"
    modelfile_path.write_text(body, encoding="utf-8")

    argv: list[str] = ["ollama", "create", tag, "-f", str(modelfile_path)]
    proc = (runner or _default_runner)(argv)
    if proc.returncode != 0:
        stderr = getattr(proc, "stderr", "") or ""
        logger.error(
            "ollama.create_failed tag=%s rc=%s err=%s",
            tag,
            proc.returncode,
            stderr,
        )
        raise OllamaModelfileError(
            f"ollama create {tag} exited {proc.returncode}: {stderr}"
        )
    return tag
