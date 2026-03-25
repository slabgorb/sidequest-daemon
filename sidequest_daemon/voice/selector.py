"""Tiered TTS engine selector — drama_weight drives engine choice."""

from sidequest_daemon.voice.protocol import SynthesisEngine

_DEFAULT_THRESHOLD = 0.5


class TieredEngineSelector:
    """Selects between low-tier and high-tier TTS engines based on drama_weight."""

    def __init__(
        self,
        low_tier: SynthesisEngine,
        high_tier: SynthesisEngine | None,
        threshold: float = _DEFAULT_THRESHOLD,
    ):
        self._low_tier = low_tier
        self._high_tier = high_tier
        self.threshold = threshold

    def _high_tier_available(self) -> bool:
        if self._high_tier is None:
            return False
        return getattr(self._high_tier, "_available", True)

    def select(self, drama_weight: float | None) -> SynthesisEngine:
        """Select engine based on drama_weight vs threshold."""
        if drama_weight is None or drama_weight < self.threshold:
            return self._low_tier
        if not self._high_tier_available():
            return self._low_tier
        return self._high_tier

    async def warm_up(self):
        """Warm up both engines."""
        await self._low_tier.warm_up()
        if self._high_tier is not None:
            await self._high_tier.warm_up()

    async def shutdown(self):
        """Shut down both engines."""
        await self._low_tier.shutdown()
        if self._high_tier is not None:
            await self._high_tier.shutdown()
