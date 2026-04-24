from sidequest_daemon.media.recipes import CameraPreset
from sidequest_daemon.renderer.models import RenderTier, StageCue


def test_tactical_sketch_removed() -> None:
    assert not hasattr(RenderTier, "TACTICAL_SKETCH")


def test_stage_cue_accepts_camera() -> None:
    cue = StageCue(
        tier=RenderTier.SCENE_ILLUSTRATION,
        subject="goblin ambush",
        camera=CameraPreset.topdown_90,
    )
    assert cue.camera is CameraPreset.topdown_90


def test_stage_cue_camera_optional() -> None:
    cue = StageCue(
        tier=RenderTier.PORTRAIT,
        subject="rux",
    )
    assert cue.camera is None
