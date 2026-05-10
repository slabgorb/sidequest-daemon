"""Integration test — exercises real FFmpeg subprocess but mocks
ACE-Step (no GPU) and R2 (no network). Catches plumbing issues that
unit tests miss: FFmpeg not installed, codec issues, file handles, etc.

NOTE: The real FFmpeg test requires libvorbis encoder (as per the
production code spec). If libvorbis is not available, that test is skipped.
The mocked fallback test always runs and verifies the pipeline wiring.
"""
import asyncio
import json
import logging
import math
import struct
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sidequest_daemon.media.ace_step_adapter import InferenceResult
from sidequest_daemon.media.music_pipeline import MusicPipeline

log = logging.getLogger(__name__)


def _has_libvorbis_encoder() -> bool:
    """Check if FFmpeg has libvorbis encoder available.

    Returns True if 'ffmpeg -codecs' lists 'libvorbis' with 'E' flag (encoder).
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-codecs"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return "libvorbis" in result.stdout and "E" in result.stdout.split("libvorbis")[0].split("\n")[-1]
    except Exception as e:
        log.warning(f"Could not check ffmpeg codecs: {e}")
        return False


def _write_sine_wav(path: Path, duration_s: int = 1, freq: float = 440.0) -> None:
    """Write a sine wave WAV at 44.1kHz, 16-bit mono.

    Args:
        path: Output file path
        duration_s: Duration in seconds
        freq: Frequency in Hz (default 440Hz = A4)
    """
    sample_rate = 44100
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / sample_rate))
            w.writeframes(struct.pack("<h", value))


@pytest.mark.skipif(
    not _has_libvorbis_encoder(),
    reason="FFmpeg missing libvorbis encoder (required by production code)"
)
@pytest.mark.asyncio
async def test_full_pipeline_with_real_ffmpeg(tmp_path):
    """Exercise the full pipeline: ACE-Step (mocked) → FFmpeg (real) → R2 (mocked).

    This catches plumbing issues:
    - FFmpeg subprocess execution
    - File handle leaks and cleanup
    - Temp directory lifecycle
    - WAV → audio format conversion

    ACE-Step and R2 are mocked because:
    - ACE-Step needs GPU/CUDA (not available in test environment)
    - R2 requires network credentials (not available in test environment)

    FFmpeg is real because:
    - We need to catch subprocess, file I/O, and conversion issues
    - The conversion is deterministic and completes quickly
    """
    # Set up a params file in genre_packs/ structure
    pack_dir = tmp_path / "genre_packs/caverns_and_claudes/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    pack_dir.mkdir(parents=True)
    json_path.write_text(json.dumps({
        "task": "text2music",
        "prompt": "intense battle music with drums and strings",
        "audio_duration": 1,
        "actual_seeds": [42],
    }))

    # Mock ACE-Step adapter: writes a real 1-second sine WAV to the output path
    def fake_adapter_run(jp, output_wav):
        _write_sine_wav(output_wav, duration_s=1)
        return InferenceResult(wav_path=output_wav, seed=42)

    adapter = MagicMock()
    adapter.run.side_effect = fake_adapter_run

    # Mock R2 uploader: records what it would upload (key, content-type, byte count)
    uploaded_bytes = []

    def capture_upload(content_bytes, r2_key, content_type):
        uploaded_bytes.append((r2_key, content_type, len(content_bytes)))
        return r2_key

    # Construct pipeline with real FFmpeg but mocked adapter and uploader
    pipeline = MusicPipeline(
        adapter=adapter,
        r2_uploader=capture_upload,
        watcher=MagicMock(),
        render_lock=asyncio.Lock(),
    )

    # Run the full pipeline. This will exercise real FFmpeg subprocess,
    # which will fail with CalledProcessError if libvorbis encoder is
    # not available in the FFmpeg binary.
    result = await pipeline.generate(json_path)

    # Verify the R2 key derivation
    assert result.r2_key == "genre_packs/caverns_and_claudes/audio/music/combat.ogg"

    # Verify exactly one upload happened with correct metadata
    assert len(uploaded_bytes) == 1
    r2_key, content_type, byte_count = uploaded_bytes[0]
    assert r2_key == "genre_packs/caverns_and_claudes/audio/music/combat.ogg"
    assert content_type == "audio/ogg"
    # A 1-second sine wave converted to audio (Vorbis or compatible codec)
    # should be at least a few hundred bytes; opus/aac/vorbis typically produce
    # 1-2KB for 1s of mono audio at quality settings comparable to q4.
    assert byte_count > 100

    # Verify result metadata
    assert result.seed == 42
    assert result.duration_ms == 1000  # 1 second = 1000ms
    assert result.elapsed_ms > 0  # Should have taken some time


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_ffmpeg_fallback(tmp_path):
    """Fallback integration test if libvorbis encoder is unavailable.

    This test uses mocked FFmpeg to verify the full pipeline wiring works
    when real FFmpeg encounters a codec issue (e.g., libvorbis not available
    in the FFmpeg binary).

    Catches:
    - Full pipeline orchestration
    - Watcher event emission
    - Temp directory cleanup
    - R2 upload integration

    Does NOT catch:
    - Actual FFmpeg subprocess behavior (mocked)
    - Codec-specific issues (mocked)
    """
    # Set up a params file
    pack_dir = tmp_path / "genre_packs/caverns_and_claudes/audio/music"
    json_path = pack_dir / "combat_input_params.json"
    pack_dir.mkdir(parents=True)
    json_path.write_text(json.dumps({
        "task": "text2music",
        "prompt": "test",
        "audio_duration": 1,
        "actual_seeds": [42],
    }))

    # Mock adapter
    def fake_adapter_run(jp, output_wav):
        _write_sine_wav(output_wav, duration_s=1)
        return InferenceResult(wav_path=output_wav, seed=42)

    adapter = MagicMock()
    adapter.run.side_effect = fake_adapter_run

    # Mock R2
    uploaded_bytes = []

    def capture_upload(content_bytes, r2_key, content_type):
        uploaded_bytes.append((r2_key, content_type, len(content_bytes)))
        return r2_key

    # Mock watcher to track events
    watcher = MagicMock()

    pipeline = MusicPipeline(
        adapter=adapter,
        r2_uploader=capture_upload,
        watcher=watcher,
        render_lock=asyncio.Lock(),
    )

    # Patch FFmpeg to simulate successful conversion
    with patch("sidequest_daemon.media.music_pipeline._run_ffmpeg") as mock_ffmpeg:
        def fake_ffmpeg(wav_path, ogg_path):
            # Simulate what FFmpeg does: read WAV, write OGG with same content size
            wav_bytes = wav_path.read_bytes()
            # OGG is typically smaller than WAV; use ~20% of original size
            ogg_path.write_bytes(b"x" * max(100, len(wav_bytes) // 5))

        mock_ffmpeg.side_effect = fake_ffmpeg
        result = await pipeline.generate(json_path)

    # Verify the full pipeline worked
    assert result.r2_key == "genre_packs/caverns_and_claudes/audio/music/combat.ogg"
    assert result.seed == 42
    assert len(uploaded_bytes) == 1

    # Verify watcher events were emitted
    event_names = [call.args[0] for call in watcher.call_args_list]
    assert "music.generation.start" in event_names
    assert "music.generation.complete" in event_names
