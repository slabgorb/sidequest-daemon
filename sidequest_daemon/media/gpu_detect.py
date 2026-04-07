"""GPU availability detection — MLX for Apple Silicon."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from opentelemetry import trace

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
    tracer = trace.get_tracer("sidequest_daemon.media.gpu_detect")
    with tracer.start_as_current_span("gpu.detect") as span:
        try:
            import mlx.core as mx
        except (ImportError, ModuleNotFoundError):
            logger.info("mlx not installed — no GPU detection possible")
            result = GpuInfo(available=False, backend="none", device_name="")
            span.set_attribute("gpu.backend", result.backend)
            span.set_attribute("gpu.available", result.available)
            return result

        try:
            device = mx.default_device()
            device_name = f"Apple Silicon ({device})"
            logger.info("MLX detected: %s", device_name)
            result = GpuInfo(available=True, backend="mlx", device_name=device_name)
            span.set_attribute("gpu.backend", result.backend)
            span.set_attribute("gpu.available", result.available)
            span.set_attribute("gpu.device_name", result.device_name)
            return result
        except Exception as exc:
            logger.warning("MLX detection failed: %s", exc)

        result = GpuInfo(available=False, backend="none", device_name="")
        span.set_attribute("gpu.backend", result.backend)
        span.set_attribute("gpu.available", result.available)
        return result
