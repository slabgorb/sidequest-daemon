import json
import pytest
from unittest.mock import MagicMock  # noqa: F401

from sidequest_daemon.media.ace_step_adapter import (  # noqa: F401
    AceStepAdapter,
    prepare_inference_params,
)


def test_prepare_inference_params_strips_output_fields(tmp_path):
    raw = {
        "task": "text2music",
        "format": "ogg",  # daemon should force this to wav
        "prompt": "test prompt",
        "lyrics": "[inst]",
        "audio_duration": 60,
        "actual_seeds": [42, 100, 200],  # only [0] preserved
        "retake_seeds": [123],  # stripped
        "timecosts": {"diffusion": 64.0},  # stripped
        "audio_path": "/Users/keithavery/stale/path.wav",  # overridden
    }
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    output_wav = tmp_path / "out.wav"

    cleaned = prepare_inference_params(json_path, output_wav)

    assert cleaned["format"] == "wav"
    assert cleaned["audio_path"] == str(output_wav)
    assert cleaned["actual_seeds"] == [42]
    assert "retake_seeds" not in cleaned
    assert "timecosts" not in cleaned
    assert cleaned["prompt"] == "test prompt"
    assert cleaned["audio_duration"] == 60


def test_prepare_inference_params_rejects_missing_seed(tmp_path):
    raw = {"task": "text2music", "prompt": "x", "audio_duration": 60}  # no actual_seeds
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="MISSING_SEED"):
        prepare_inference_params(json_path, tmp_path / "out.wav")


def test_prepare_inference_params_rejects_empty_seed_list(tmp_path):
    raw = {"task": "text2music", "actual_seeds": []}
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="MISSING_SEED"):
        prepare_inference_params(json_path, tmp_path / "out.wav")


def test_prepare_inference_params_rejects_non_integer_seed(tmp_path):
    raw = {"task": "text2music", "actual_seeds": ["abc"]}
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="MISSING_SEED"):
        prepare_inference_params(json_path, tmp_path / "out.wav")
