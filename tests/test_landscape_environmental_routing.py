"""Regression tests for playtest 2026-04-30 — `kind=poi without scope`
onion peel after daemon PR #63.

Pre-fix flow:
1. Server narrator emitted ``tier=landscape`` with prose subject
   ("Cramped wrench-house galley under a coolant pipe...").
2. PR #63 ensured the tier survived to the daemon-side build_render_target.
3. ``build_render_target`` mapped LANDSCAPE → ``kind=poi, place=cue.subject``.
4. The pydantic validator's poi guard rejected the prose subject because
   it didn't carry a ``where:<world>/<slug>`` scheme — "poi targets must
   reference a specific place in world 'coyote_reach'; got scope ''".
5. COMPOSE_FAILED.

Fix:
- LANDSCAPE branch in ``build_render_target`` discriminates by the
  ``where:`` scheme. Explicit-POI renders (subject starting with
  ``where:``) keep routing to ``kind=poi``. Prose subjects route to
  ``kind=illustration`` with empty participants.
- ``RenderTarget`` validator relaxed: ``kind=illustration`` no longer
  requires non-empty ``participants``. Empty participants is the
  environmental-scene shape; action prose + ART_SENSIBILITY layers
  carry the visual.
"""

from __future__ import annotations

import pytest

from sidequest_daemon.media.recipes import CameraPreset, RenderTarget
from sidequest_daemon.media.workers.zimage_mlx_worker import (
    build_render_target,
)
from sidequest_daemon.renderer.models import RenderTier, StageCue


def _cue(*, tier: RenderTier, subject: str, characters: list[str] | None = None,
         location: str = "") -> StageCue:
    return StageCue(
        tier=tier,
        subject=subject,
        characters=characters or [],
        location=location,
        metadata={"world": "coyote_reach", "genre": "space_opera"},
    )


def test_landscape_with_prose_subject_routes_to_environmental_illustration():
    """The Parsley turn-4 reproducer: prose subject, no PCs, no place
    ref. Pre-fix: COMPOSE_FAILED on ``kind=poi`` scope check. Post-fix:
    valid ``kind=illustration`` with empty participants.
    """
    cue = _cue(
        tier=RenderTier.LANDSCAPE,
        subject="Cramped wrench-house galley under a coolant pipe, one amber light strip",
    )
    target = build_render_target(cue)
    assert target.kind == "illustration", (
        f"prose-subject LANDSCAPE must route to environmental "
        f"illustration, got kind={target.kind!r}"
    )
    assert target.participants == []
    assert target.action.startswith("Cramped wrench-house galley")
    assert target.camera == CameraPreset.scene


def test_landscape_with_where_scheme_subject_still_routes_to_poi():
    """Explicit POI renders must keep routing to ``kind=poi`` — the
    content-tool / poi-pregeneration path.
    """
    cue = _cue(
        tier=RenderTier.LANDSCAPE,
        subject="where:coyote_reach/far_landing",
    )
    target = build_render_target(cue)
    assert target.kind == "poi", (
        f"where:-scheme LANDSCAPE must keep routing to poi, got "
        f"kind={target.kind!r}"
    )
    assert target.place == "where:coyote_reach/far_landing"


def test_environmental_illustration_validator_accepts_empty_participants():
    """The schema-level relaxation: empty participants is the valid
    environmental-scene shape. Action and camera are still required.
    """
    target = RenderTarget(
        kind="illustration",
        world="coyote_reach",
        genre="space_opera",
        participants=[],
        action="A breaching corridor at dawn",
        camera=CameraPreset.scene,
    )
    assert target.participants == []
    assert target.action == "A breaching corridor at dawn"


def test_illustration_validator_still_requires_action():
    """Backward-compat: action remains mandatory."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RenderTarget(
            kind="illustration",
            world="coyote_reach",
            genre="space_opera",
            participants=[],
            action="",
            camera=CameraPreset.scene,
        )


def test_illustration_validator_still_requires_camera():
    """Backward-compat: camera remains mandatory."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RenderTarget(
            kind="illustration",
            world="coyote_reach",
            genre="space_opera",
            participants=[],
            action="A scene",
            camera=None,
        )


def test_illustration_with_participants_still_works_for_pc_scenes():
    """Forward-compat: PC-scene illustrations are unchanged. The relaxed
    validator only adds an environmental shape; it doesn't subtract.
    """
    target = RenderTarget(
        kind="illustration",
        world="coyote_reach",
        genre="space_opera",
        participants=["pc:parsley"],
        action="Parsley negotiates with the Compact officers",
        camera=CameraPreset.scene,
    )
    assert target.participants == ["pc:parsley"]
