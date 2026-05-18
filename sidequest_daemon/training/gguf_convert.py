"""MLX safetensors -> GGUF LoRA conversion for story 48-3 substep (c).

The subprocess invocation is dependency-injected (mirrors ``trainer.py``'s
``trainer_fn``) so the CI layer can assert argv construction and failure
handling without llama.cpp present. The real conversion is operator-evidence
only and runs on Keith's M3 Ultra.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

ADAPTER_FILE = "adapters.safetensors"
CONVERT_SCRIPT = "convert_lora_to_gguf.py"
LLAMA_CPP_DIR_ENV = "SIDEQUEST_LLAMA_CPP_DIR"

# subprocess.CompletedProcess is the structural contract; only .returncode
# (and optionally .stderr) is read, so any compatible object works.
Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class GgufConversionError(RuntimeError):
    """Raised when MLX -> GGUF conversion fails or its input is absent."""


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Operator-path runner: resolve llama.cpp's converter and run it.

    Fails loudly (No Silent Fallbacks) if llama.cpp is not configured —
    this path is exercised only on the operator host, never in CI.
    """
    llama_dir = os.environ.get(LLAMA_CPP_DIR_ENV)
    if not llama_dir:
        raise FileNotFoundError(
            f"{LLAMA_CPP_DIR_ENV} is unset; cannot locate {CONVERT_SCRIPT}. "
            f"GGUF conversion is operator-evidence only (M3 Ultra)."
        )
    script = Path(llama_dir) / CONVERT_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(f"llama.cpp converter not found at {script}")
    resolved = [*argv]
    resolved[1] = str(script)
    return subprocess.run(resolved, capture_output=True, text=True)  # noqa: S603


def convert_lora_to_gguf(
    adapter_dir: Path,
    out_path: Path,
    *,
    runner: Runner | None = None,
) -> Path:
    """Convert ``adapter_dir/adapters.safetensors`` to ``out_path`` (GGUF).

    Returns ``out_path`` on success. Raises :class:`GgufConversionError`
    if the input adapter is missing or the converter exits non-zero.
    """
    safetensors = Path(adapter_dir) / ADAPTER_FILE
    if not safetensors.is_file():
        raise GgufConversionError(
            f"no {ADAPTER_FILE} in {adapter_dir}; nothing to convert"
        )

    argv: list[str] = [
        sys.executable,
        CONVERT_SCRIPT,
        "--outfile",
        str(out_path),
        str(safetensors),
    ]

    proc = (runner or _default_runner)(argv)
    if proc.returncode != 0:
        stderr = getattr(proc, "stderr", "") or ""
        logger.error("gguf.convert_failed rc=%s err=%s", proc.returncode, stderr)
        raise GgufConversionError(
            f"convert_lora_to_gguf exited {proc.returncode}: {stderr}"
        )
    return Path(out_path)
