"""Recipe-driven prompt composition types."""

from __future__ import annotations

from enum import Enum


class Slot(str, Enum):
    """Named slots that every recipe declares."""

    CASTING = "casting"
    LOCATION = "location"
    DIRECTION_ACTION = "direction_action"
    DIRECTION_CAMERA = "direction_camera"
    ART_SENSIBILITY = "art_sensibility"


class LOD(str, Enum):
    """Character level-of-detail for prompt contribution."""

    SOLO = "solo"
    LONG = "long"
    SHORT = "short"
    BACKGROUND = "background"


class PlaceLOD(str, Enum):
    """Place level-of-detail — subject or setting."""

    SOLO = "solo"
    BACKDROP = "backdrop"


class CameraPreset(str, Enum):
    """Enumerated camera presets — 17 total, stills-only (no motion)."""

    # Portrait framings
    portrait_3q = "portrait_3q"
    portrait_profile = "portrait_profile"
    portrait_closeup = "portrait_closeup"
    portrait_full_body = "portrait_full_body"
    # POI framings
    wide_establishing = "wide_establishing"
    low_angle_hero = "low_angle_hero"
    interior_wide = "interior_wide"
    aerial_oblique = "aerial_oblique"
    # Illustration framings
    scene = "scene"
    over_shoulder = "over_shoulder"
    wide_action = "wide_action"
    closeup_action = "closeup_action"
    topdown_90 = "topdown_90"
    # Signature shots
    extreme_closeup_leone = "extreme_closeup_leone"
    dutch_tilt = "dutch_tilt"
    single_point_perspective_kubrick = "single_point_perspective_kubrick"
    trunk_shot_tarantino = "trunk_shot_tarantino"
