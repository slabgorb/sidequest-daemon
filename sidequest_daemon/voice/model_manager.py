"""Voice model auto-download and cache management — Piper and Kokoro."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from sidequest_daemon.voice.errors import SynthesisError

logger = logging.getLogger(__name__)

PIPER_RELEASE_BASE = (
    "https://github.com/rhasspy/piper/releases/download/v0.0.2"
)


class ModelDownloadError(SynthesisError):
    """Failed to download a Piper voice model."""

    def __init__(self, message: str, *, model_name: str = "") -> None:
        super().__init__(message)
        self.model_name = model_name


class PiperModelManager:
    """Download, cache, and validate Piper TTS voice models."""

    DEFAULT_CACHE_DIR = Path.home() / ".sidequest" / "models" / "piper"

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    # -- Name validation -----------------------------------------------------

    def validate_model_name(self, name: str) -> None:
        """Raise ValueError if model name is invalid or contains path traversal."""
        if not name:
            raise ValueError("invalid model name: name must not be empty")
        if "/" in name or "\\" in name:
            raise ValueError(f"invalid model name: slashes not allowed: {name!r}")
        if ".." in name:
            raise ValueError(f"invalid model name: path traversal not allowed: {name!r}")

    # -- Path helpers --------------------------------------------------------

    def model_path(self, name: str) -> Path:
        """Return the expected path of the .onnx file for a model."""
        return self.cache_dir / name / f"{name}.onnx"

    # -- Cache checking ------------------------------------------------------

    def is_cached(self, name: str) -> bool:
        """Check if a model is fully cached (non-empty .onnx + .onnx.json)."""
        onnx = self.cache_dir / name / f"{name}.onnx"
        json_cfg = self.cache_dir / name / f"{name}.onnx.json"
        return onnx.exists() and onnx.stat().st_size > 0 and json_cfg.exists()

    def validate_model(self, name: str) -> bool:
        """Validate a cached model: onnx non-empty and json is parseable."""
        onnx = self.cache_dir / name / f"{name}.onnx"
        json_cfg = self.cache_dir / name / f"{name}.onnx.json"
        if not onnx.exists() or onnx.stat().st_size == 0:
            return False
        if not json_cfg.exists():
            return False
        try:
            json.loads(json_cfg.read_text())
        except (json.JSONDecodeError, ValueError):
            return False
        return True

    def list_cached(self) -> list[str]:
        """Return names of all valid cached models."""
        if not self.cache_dir.exists():
            return []
        result = []
        for d in sorted(self.cache_dir.iterdir()):
            if d.is_dir() and self.is_cached(d.name):
                result.append(d.name)
        return result

    def find_missing_models(self, names: list[str]) -> list[str]:
        """Return model names from the list that are not cached."""
        return [n for n in names if not self.is_cached(n)]

    # -- Download ------------------------------------------------------------

    async def _fetch_file(self, url: str) -> bytes:
        """Fetch a file from a URL. Overridden in tests via patch."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()

    async def download_model(self, name: str) -> None:
        """Download a Piper model (.onnx + .onnx.json) into the cache."""
        self.validate_model_name(name)

        if self.is_cached(name):
            return

        logger.info("Downloading Piper model %s...", name)

        model_dir = self.cache_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        onnx_url = f"{PIPER_RELEASE_BASE}/{name}.onnx"
        json_url = f"{PIPER_RELEASE_BASE}/{name}.onnx.json"

        try:
            onnx_data = await self._fetch_file(onnx_url)
            json_data = await self._fetch_file(json_url)

            onnx_path = model_dir / f"{name}.onnx"
            json_path = model_dir / f"{name}.onnx.json"

            with open(onnx_path, "wb") as f:
                f.write(onnx_data)
            with open(json_path, "wb") as f:
                f.write(json_data)
        except ModelDownloadError:
            raise
        except Exception as exc:
            # Clean up partial downloads
            for f in model_dir.iterdir():
                f.unlink(missing_ok=True)
            if model_dir.exists() and not any(model_dir.iterdir()):
                shutil.rmtree(model_dir, ignore_errors=True)
            raise ModelDownloadError(
                f"Failed to download model '{name}': {exc}. "
                f"You can manually download it from {PIPER_RELEASE_BASE}/ "
                f"and place the files in {model_dir}",
                model_name=name,
            ) from exc

        logger.info("Download complete: %s cached in %s", name, model_dir)

    async def ensure_model(self, name: str) -> None:
        """Ensure a model is cached, downloading if necessary. Deduplicates concurrent calls."""
        if self.is_cached(name):
            return

        async with self._global_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            lock = self._locks[name]

        async with lock:
            if self.is_cached(name):
                return
            try:
                await self.download_model(name)
            except ModelDownloadError:
                logger.warning("Failed to download model %s", name)
                raise

    async def ensure_all_models(self, names: list[str]) -> None:
        """Ensure all listed models are cached."""
        for name in names:
            await self.download_model(name)


# ---------------------------------------------------------------------------
# Kokoro model manager
# ---------------------------------------------------------------------------

KOKORO_HF_BASE = (
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main"
)


class KokoroModelManager:
    """Download, cache, and validate Kokoro TTS voice models."""

    DEFAULT_CACHE_DIR = Path.home() / ".sidequest" / "models" / "kokoro"

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    # -- Name validation -----------------------------------------------------

    def validate_model_name(self, name: str) -> None:
        """Raise ValueError if model name is invalid or contains path traversal."""
        if not name:
            raise ValueError("invalid model name: name must not be empty")
        if "/" in name or "\\" in name:
            raise ValueError(f"invalid model name: slashes not allowed: {name!r}")
        if ".." in name:
            raise ValueError(f"invalid model name: path traversal not allowed: {name!r}")

    # -- Path helpers --------------------------------------------------------

    def model_dir(self, name: str) -> Path:
        """Return the directory for a model version."""
        return self.cache_dir / name

    def model_path(self, name: str) -> Path:
        """Return the expected path of the .pth weights file for a model."""
        return self.cache_dir / name / f"{name}.pth"

    def voice_dir(self, name: str) -> Path:
        """Return the voices subdirectory for a model."""
        return self.cache_dir / name / "voices"

    def voice_path(self, name: str, voice_name: str) -> Path:
        """Return the path to a specific voice .pt file."""
        return self.cache_dir / name / "voices" / f"{voice_name}.pt"

    # -- Cache checking ------------------------------------------------------

    def is_model_cached(self, name: str) -> bool:
        """Check if a model is fully cached (non-empty .pth + config.json)."""
        pth = self.cache_dir / name / f"{name}.pth"
        cfg = self.cache_dir / name / "config.json"
        return pth.exists() and pth.stat().st_size > 0 and cfg.exists()

    def is_voice_cached(self, name: str, voice_name: str) -> bool:
        """Check if a voice .pt file is cached and non-empty."""
        vpath = self.voice_path(name, voice_name)
        return vpath.exists() and vpath.stat().st_size > 0

    def validate_model(self, name: str) -> bool:
        """Validate a cached model: pth non-empty and config.json is parseable."""
        pth = self.cache_dir / name / f"{name}.pth"
        cfg = self.cache_dir / name / "config.json"
        if not pth.exists() or pth.stat().st_size == 0:
            return False
        if not cfg.exists():
            return False
        try:
            json.loads(cfg.read_text())
        except (json.JSONDecodeError, ValueError):
            return False
        return True

    def list_cached_voices(self, name: str) -> list[str]:
        """Return names of all valid cached voice .pt files for a model."""
        vdir = self.voice_dir(name)
        if not vdir.exists():
            return []
        result = []
        for f in sorted(vdir.iterdir()):
            if f.suffix == ".pt" and f.stat().st_size > 0:
                result.append(f.stem)
        return result

    def find_missing_voices(self, name: str, voice_names: list[str]) -> list[str]:
        """Return voice names from the list that are not cached."""
        return [v for v in voice_names if not self.is_voice_cached(name, v)]

    # -- Download ------------------------------------------------------------

    async def _download_from_hf(self, url: str) -> bytes:
        """Fetch a file from HuggingFace. Overridden in tests via patch."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()

    async def download_model(self, name: str) -> None:
        """Download a Kokoro model (.pth + config.json) into the cache."""
        self.validate_model_name(name)

        if self.is_model_cached(name):
            return

        logger.info("Downloading Kokoro model %s...", name)

        model_dir = self.cache_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        pth_url = f"{KOKORO_HF_BASE}/{name}.pth"
        config_url = f"{KOKORO_HF_BASE}/config.json"

        try:
            pth_data = await self._download_from_hf(pth_url)
            config_data = await self._download_from_hf(config_url)

            pth_path = model_dir / f"{name}.pth"
            config_path = model_dir / "config.json"

            with open(pth_path, "wb") as f:
                f.write(pth_data)
            with open(config_path, "wb") as f:
                f.write(config_data)
        except ModelDownloadError:
            raise
        except Exception as exc:
            # Clean up partial downloads
            if model_dir.exists():
                for f in model_dir.iterdir():
                    f.unlink(missing_ok=True)
                shutil.rmtree(model_dir, ignore_errors=True)
            raise ModelDownloadError(
                f"Failed to download model '{name}': {exc}. "
                f"You can manually download it from {KOKORO_HF_BASE}/ "
                f"and place the files in {model_dir}",
                model_name=name,
            ) from exc

        logger.info("Download complete: %s cached in %s", name, model_dir)

    async def download_voice(self, name: str, voice_name: str) -> None:
        """Download a single voice .pt file into the cache."""
        if self.is_voice_cached(name, voice_name):
            return

        logger.info("Downloading Kokoro voice %s for model %s...", voice_name, name)

        voices_dir = self.voice_dir(name)
        voices_dir.mkdir(parents=True, exist_ok=True)

        url = f"{KOKORO_HF_BASE}/voices/{voice_name}.pt"

        try:
            data = await self._download_from_hf(url)

            voice_file = voices_dir / f"{voice_name}.pt"
            with open(voice_file, "wb") as f:
                f.write(data)
        except ModelDownloadError:
            raise
        except Exception as exc:
            # Clean up partial file
            partial = voices_dir / f"{voice_name}.pt"
            if partial.exists():
                partial.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Failed to download voice '{voice_name}': {exc}. "
                f"You can manually download it from {KOKORO_HF_BASE}/voices/ "
                f"and place it in {voices_dir}",
                model_name=voice_name,
            ) from exc

        logger.info("Voice download complete: %s", voice_name)

    async def download_voices(self, name: str, voice_names: list[str]) -> None:
        """Download multiple voice files."""
        for voice_name in voice_names:
            await self.download_voice(name, voice_name)

    async def ensure_model(self, name: str) -> None:
        """Ensure a model is cached, downloading if necessary. Deduplicates concurrent calls."""
        if self.is_model_cached(name):
            return

        async with self._global_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            lock = self._locks[name]

        async with lock:
            if self.is_model_cached(name):
                return
            try:
                await self.download_model(name)
            except ModelDownloadError:
                logger.warning("Failed to download model %s", name)
                raise

    async def ensure_voices(self, name: str, voice_names: list[str]) -> None:
        """Ensure all listed voices are cached, downloading missing ones."""
        missing = self.find_missing_voices(name, voice_names)
        for voice_name in missing:
            await self.download_voice(name, voice_name)

    async def ensure_ready(
        self, name: str, *, voices: list[str] | None = None
    ) -> None:
        """High-level entry point: ensure model and optionally voices are ready."""
        await self.ensure_model(name)
        if voices is not None:
            await self.ensure_voices(name, voices)
