"""ACE-Step music generation worker for the unified daemon.

Story 23-13: Wraps ACE-Step pipeline as a daemon worker with the same
interface as FluxWorker (load_model, warm_up, render, cleanup).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path


class ACEStepWorker:
    """ACE-Step music generation worker."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None

    def load_model(self) -> None:
        """Load the ACE-Step pipeline.

        Requires ACE_STEP_PATH env var pointing to the ACE-Step installation.
        """
        import os
        import sys

        ace_step_path = os.environ.get("ACE_STEP_PATH", "")
        if not ace_step_path:
            raise RuntimeError(
                "ACE_STEP_PATH environment variable is not set. "
                "Set it to your ACE-Step installation directory."
            )
        sys.path.insert(0, ace_step_path)

        # ACE-Step's dependencies (loguru, etc.) live in its own venv
        ace_venv_site = Path(ace_step_path) / ".venv" / "lib"
        for pydir in sorted(ace_venv_site.glob("python*/site-packages")):
            if str(pydir) not in sys.path:
                sys.path.insert(1, str(pydir))

        from acestep.pipeline_ace_step import ACEStepPipeline

        self.pipeline = ACEStepPipeline(
            checkpoint_dir=ace_step_path,
            dtype="float32",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False,
        )

    def warm_up(self) -> dict:
        """Warm up the ACE-Step pipeline. Returns timing metadata."""
        start = time.monotonic()
        # ACE-Step doesn't need a dummy generation for warmup —
        # model loading is the expensive part
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {"warmup_ms": elapsed_ms}

    def render(self, params: dict) -> dict:
        """Generate music from params. Returns result dict with audio_path."""
        prompt = params.get("prompt", "background music")
        duration = params.get("duration", 60)
        seed = params.get("seed", 42)

        output_name = f"music_{uuid.uuid4().hex[:8]}.wav"
        output_path = self.output_dir / output_name

        start = time.monotonic()
        self.pipeline(
            audio_duration=duration,
            prompt=prompt,
            lyrics="[inst]",
            infer_step=60,
            guidance_scale=15,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10,
            manual_seeds=str(seed),
            guidance_interval=0.5,
            guidance_interval_decay=0,
            min_guidance_scale=3,
            use_erg_tag=True,
            use_erg_lyric=False,
            use_erg_diffusion=True,
            oss_steps="",
            guidance_scale_text=0.0,
            guidance_scale_lyric=0.0,
            save_path=str(output_path),
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        return {
            "audio_path": str(output_path),
            "duration_ms": duration * 1000,
            "elapsed_ms": elapsed_ms,
        }

    def cleanup(self) -> None:
        """Release the pipeline."""
        self.pipeline = None
