"""Recipe-driven prompt composition types."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator


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


class RenderTarget(BaseModel):
    """The composable render input. One type serves all three kinds."""

    kind: Literal["portrait", "poi", "illustration"]
    world: str
    genre: str

    # Portrait
    character: str | None = None
    pose_override: str | None = None
    background: str | None = None  # optional where:<scope>/<slug>

    # POI
    place: str | None = None  # where:<world>/<slug> — specific only

    # Illustration
    participants: list[str] = []
    location: str | None = None  # where:<scope>/<slug> — specific or archetypal
    action: str = ""
    camera: CameraPreset | None = None

    # Debug/preview only
    lod_override: dict[str, LOD] | None = None

    @model_validator(mode="after")
    def _enforce_kind_shape(self) -> "RenderTarget":
        if self.kind == "portrait":
            if self.character is None:
                raise ValueError("portrait targets require `character`")
            if self.place or self.participants or self.action:
                raise ValueError(
                    "portrait targets must not set place/participants/action",
                )
        elif self.kind == "poi":
            if self.place is None:
                raise ValueError("poi targets require `place`")
            # Specific-place guard: the scope segment must match `world`.
            # Full validation (catalog lookup) is enforced by the composer;
            # this guard rejects obviously-archetypal refs at the schema level.
            _, _, scope_slug = self.place.partition(":")
            scope = scope_slug.split("/", 1)[0]
            if scope != self.world:
                raise ValueError(
                    f"poi targets must reference a specific place in world "
                    f"{self.world!r}; got scope {scope!r}",
                )
            if self.character or self.participants or self.action:
                raise ValueError(
                    "poi targets must not set character/participants/action",
                )
        elif self.kind == "illustration":
            if not self.participants:
                raise ValueError("illustration targets require `participants`")
            if not self.action:
                raise ValueError("illustration targets require `action`")
            if not self.location:
                raise ValueError("illustration targets require `location`")
            if self.camera is None:
                raise ValueError("illustration targets require `camera`")
        return self


class Recipe(BaseModel):
    """A canonical recipe — names the source bindings for each slot."""

    kind: Literal["portrait", "poi", "illustration"]
    casting: str
    location: str
    direction_action: str
    direction_camera: str  # a CameraPreset member name, or "{camera}" to
                           # pull from RenderTarget.camera
    art_sensibility: list[str]  # ordered cascade: e.g. ["GENRE","WORLD","CULTURE"]


class LayerContribution(BaseModel):
    slot: str
    source: str
    tokens: str
    estimated_tokens: int


class ComposedPrompt(BaseModel):
    positive_prompt: str
    clip_prompt: str
    negative_prompt: str
    worker_type: str
    seed: int
    layers: list[LayerContribution]
    dropped_layers: list[str]
    warnings: list[str]
