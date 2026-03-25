"""Exception hierarchy for voice synthesis errors."""

from __future__ import annotations


class SynthesisError(Exception):
    """Base exception for all voice synthesis errors."""


class ModelLoadError(SynthesisError):
    """TTS model failed to load."""


class SynthesisFailedError(SynthesisError):
    """Synthesis operation failed."""


class UnsupportedLanguageError(SynthesisError):
    """Requested language is not supported by the engine."""


class TextNormalizationError(SynthesisError):
    """Text normalization failed."""


class VoicePresetError(SynthesisError):
    """Invalid or unknown voice preset."""
