"""Exception hierarchy for voice synthesis errors."""

from __future__ import annotations


class SynthesisError(Exception):
    """Base exception for all voice synthesis errors."""


class ModelLoadError(SynthesisError):
    """TTS model failed to load."""


class SynthesisFailedError(SynthesisError):
    """Synthesis operation failed."""


class VoicePresetError(SynthesisError):
    """Invalid or unknown voice preset."""
