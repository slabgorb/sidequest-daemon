"""GPU availability detection — MLX for Apple Silicon."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

GpuBackend = Literal["mlx", "none"]


@dataclass
class GpuInfo:
    """GPU detection result."""

    available: bool
    backend: GpuBackend
    device_name: str


def detect_gpu() -> GpuInfo:
    """Detect GPU availability via MLX on Apple Silicon."""
    try:
        import mlx.core as mx
    except (ImportError, ModuleNotFoundError):
        logger.info("mlx not installed — no GPU detection possible")
        return GpuInfo(available=False, backend="none", device_name="")

    try:
        device = mx.default_device()
        device_name = f"Apple Silicon ({device})"
        logger.info("MLX detected: %s", device_name)
        return GpuInfo(available=True, backend="mlx", device_name=device_name)
    except Exception as exc:
        logger.warning("MLX detection failed: %s", exc)

    return GpuInfo(available=False, backend="none", device_name="")
