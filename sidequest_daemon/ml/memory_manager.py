"""ModelMemoryManager — GPU memory budget coordinator for ML backends.

Story 5-10: ModelMemoryManager — 80GB shared ML budget.
Coordinates GPU memory across multiple ML backends on M3 Max unified memory.
"""

from __future__ import annotations

import time

from sidequest_daemon.audio.protocol import AudioBackend

_80_GB = 80 * 1024 * 1024 * 1024


class ModelMemoryManager:
    """Manages a shared GPU memory budget across registered ML backends.

    Uses LRU eviction to ensure total loaded model memory stays within budget.
    """

    def __init__(self, budget_bytes: int = _80_GB) -> None:
        self._budget_bytes = budget_bytes
        self._backends: dict[str, AudioBackend] = {}
        self._last_used: dict[str, float | None] = {}

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def total_usage(self) -> int:
        return sum(b.gpu_memory_bytes for b in self._backends.values())

    @property
    def available(self) -> int:
        return self._budget_bytes - self.total_usage

    def register(self, backend: AudioBackend) -> None:
        if backend.name not in self._backends:
            self._backends[backend.name] = backend
            self._last_used[backend.name] = None

    def unregister(self, backend: AudioBackend) -> None:
        self._backends.pop(backend.name, None)
        self._last_used.pop(backend.name, None)

    async def ensure_loaded(self, backend: AudioBackend) -> None:
        if backend.is_ready:
            self._last_used[backend.name] = time.monotonic()
            return

        needed = (
            backend._memory_bytes
            if hasattr(backend, "_memory_bytes")
            else self._estimate_memory(backend)
        )

        if needed > self._budget_bytes:
            raise MemoryError(
                f"Backend '{backend.name}' requires {needed} bytes "
                f"but total budget is {self._budget_bytes} bytes"
            )

        # Check if evicting everything (except target) would free enough
        evictable = sum(
            b.gpu_memory_bytes
            for name, b in self._backends.items()
            if name != backend.name and b.is_ready
        )
        if self.total_usage + needed - evictable > self._budget_bytes:
            raise MemoryError(
                f"Backend '{backend.name}' requires {needed} bytes "
                f"but cannot free enough memory even after full eviction"
            )

        # Evict LRU backends until there's room
        while self.total_usage + needed > self._budget_bytes:
            await self._evict_lru(exclude=backend.name)

        await backend.warm_up()
        self._last_used[backend.name] = time.monotonic()

    async def _evict_lru(self, exclude: str) -> None:
        candidates = [
            (name, b)
            for name, b in self._backends.items()
            if name != exclude and b.is_ready and self._last_used.get(name) is not None
        ]
        if not candidates:
            raise MemoryError("No evictable backends available")

        lru_name = min(candidates, key=lambda x: self._last_used[x[0]])[0]
        await self._backends[lru_name].shutdown()

    def _estimate_memory(self, backend: AudioBackend) -> int:
        """Estimate memory for backends without _memory_bytes attr."""
        from sidequest_daemon.audio.musicgen_backend import _MODEL_MEMORY_BYTES

        if backend.name == "musicgen":
            return _MODEL_MEMORY_BYTES
        return 0

    def status(self) -> list[dict]:
        return [
            {
                "name": name,
                "memory": b.gpu_memory_bytes,
                "is_ready": b.is_ready,
                "last_used": self._last_used.get(name),
            }
            for name, b in self._backends.items()
        ]
