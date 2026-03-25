"""GPU availability detection — CUDA → MPS → none fallback chain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

GpuBackend = Literal["cuda", "mps", "none"]


@dataclass
class GpuInfo:
    """GPU detection result."""

    available: bool
    backend: GpuBackend
    device_name: str


def detect_gpu() -> GpuInfo:
    """Detect GPU availability: CUDA first, then MPS, then none."""
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        logger.info("torch not installed — no GPU detection possible")
        return GpuInfo(available=False, backend="none", device_name="")

    # torch may be None in sys.modules (patched out)
    if torch is None:
        return GpuInfo(available=False, backend="none", device_name="")

    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s", device_name)
            return GpuInfo(available=True, backend="cuda", device_name=device_name)

        if torch.backends.mps.is_available():
            logger.info("Apple MPS detected")
            return GpuInfo(available=True, backend="mps", device_name="Apple Silicon")

    except Exception as exc:
        logger.warning("GPU detection failed: %s", exc)

    return GpuInfo(available=False, backend="none", device_name="")
