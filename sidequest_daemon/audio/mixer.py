"""AudioMixer — pygame.mixer backend with 3 named channels, crossfade, ducking.

Story 5-3: AudioMixer — pygame.mixer, 3 channels, crossfade, ducking
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidequest_daemon.audio.models import MoodCategory


@dataclass
class MixerSnapshot:
    """Frozen snapshot of AudioMixer state."""

    channel_volumes: dict[str, float]
    channel_tracks: dict[str, Path | None]
    channel_playing: dict[str, bool]
    channel_muted: dict[str, bool]
    master_volume: float
    ducked: bool
    mood: MoodCategory | None
    tts_state: str | None = None


CHANNEL_NAMES = ("music", "sfx", "ambience")
_CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNEL_NAMES)}
_FADE_CHANNEL_IDX = len(CHANNEL_NAMES)  # Extra channel for crossfade overlap
_FADE_TICK_S = 0.016  # ~60 fps


class AudioMixer:
    """Three-channel audio mixer backed by pygame.mixer."""

    def __init__(
        self,
        *,
        ducking_enabled: bool = True,
        duck_level: float = 0.3,
        duck_attack_ms: int = 200,
        duck_release_ms: int = 500,
    ) -> None:
        import pygame.mixer

        self._mixer = pygame.mixer
        self._mixer.init()

        self.channels: dict[str, object] = {
            name: self._mixer.Channel(idx) for name, idx in _CHANNEL_INDICES.items()
        }

        self.ducking_enabled = ducking_enabled
        self.duck_level = duck_level
        self.duck_attack_ms = duck_attack_ms
        self.duck_release_ms = duck_release_ms

        self._volumes: dict[str, float] = {name: 1.0 for name in CHANNEL_NAMES}
        self._muted: dict[str, bool] = {name: False for name in CHANNEL_NAMES}
        self._master_volume: float = 1.0
        self._playing: dict[str, bool] = {name: False for name in CHANNEL_NAMES}
        self._current_track: dict[str, Path | None] = {
            name: None for name in CHANNEL_NAMES
        }
        self._ducked: bool = False
        self._shutdown: bool = False
        self.crossfade_duration_ms: int = 3000
        self.fade_state: dict | None = None
        self._current_sound: dict[str, object | None] = {
            name: None for name in CHANNEL_NAMES
        }

        # Extra channels: fade overlap + dedicated TTS
        self._mixer.set_num_channels(max(len(CHANNEL_NAMES) + 2, 5))
        self._fade_channel = self._mixer.Channel(_FADE_CHANNEL_IDX)
        self.tts_channel = self._mixer.Channel(_FADE_CHANNEL_IDX + 1)

        # Background timer for auto-advancing fades
        self._fade_timer: threading.Timer | None = None
        self._fade_timer_ts: float = 0.0

    # -- validation helper --------------------------------------------------

    def _require_channel(self, channel: str) -> None:
        if channel not in self.channels:
            raise KeyError(channel)

    # -- volume helpers -----------------------------------------------------

    @property
    def master_volume(self) -> float:
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        self._master_volume = max(0.0, min(1.0, value))
        self._apply_volumes()

    def get_volume(self, channel: str) -> float:
        self._require_channel(channel)
        return self._volumes[channel]

    def set_volume(self, channel: str, volume: float) -> None:
        self._require_channel(channel)
        self._volumes[channel] = max(0.0, min(1.0, volume))
        self._apply_volumes()

    def get_effective_volume(self, channel: str) -> float:
        self._require_channel(channel)
        if self._muted[channel]:
            return 0.0
        vol = self._volumes[channel] * self._master_volume
        if self._ducked and channel in ("music", "ambience"):
            vol = self.duck_level
        return vol

    def _apply_volumes(self) -> None:
        for name in CHANNEL_NAMES:
            self.channels[name].set_volume(self.get_effective_volume(name))

    # -- mute / unmute ------------------------------------------------------

    def is_muted(self, channel: str) -> bool:
        self._require_channel(channel)
        return self._muted[channel]

    def mute(self, channel: str) -> None:
        self._require_channel(channel)
        self._muted[channel] = True
        self._apply_volumes()

    def unmute(self, channel: str) -> None:
        self._require_channel(channel)
        self._muted[channel] = False
        self._apply_volumes()

    # -- playback -----------------------------------------------------------

    def play(
        self,
        *,
        channel: str,
        path: Path,
        loop: bool = False,
        fade_in_ms: int = 0,
    ) -> None:
        self._require_channel(channel)
        sound = self._mixer.Sound(str(path))
        loops = -1 if loop else 0
        self.channels[channel].play(sound, loops=loops)
        self._playing[channel] = True
        self._current_track[channel] = path
        self._current_sound[channel] = sound

        if fade_in_ms > 0:
            self._volumes[channel] = 0.0
            self.fade_state = {
                "direction": "in",
                "channel": channel,
                "duration_ms": fade_in_ms,
                "elapsed_ms": 0,
            }
            self._apply_volumes()
            self._start_fade_timer()

        if channel == "sfx" and self.ducking_enabled:
            self._ducked = True
            self._apply_volumes()

    def stop(self, channel: str, fade_out_ms: int = 0) -> None:
        self._require_channel(channel)
        if fade_out_ms > 0:
            self.fade_state = {
                "direction": "out",
                "channel": channel,
                "duration_ms": fade_out_ms,
                "elapsed_ms": 0,
            }
            self._start_fade_timer()
            return
        self.channels[channel].stop()
        self._playing[channel] = False
        self._current_track[channel] = None
        self._current_sound[channel] = None

    def stop_all(self) -> None:
        for name in CHANNEL_NAMES:
            self.stop(name)

    def is_playing(self, channel: str) -> bool:
        self._require_channel(channel)
        return self._playing[channel]

    def current_track(self, channel: str) -> Path | None:
        self._require_channel(channel)
        return self._current_track[channel]

    # -- crossfade ----------------------------------------------------------

    def crossfade(
        self,
        *,
        channel: str,
        path: Path,
        duration_ms: int = 3000,
    ) -> None:
        self._require_channel(channel)
        self.crossfade_duration_ms = duration_ms

        if duration_ms == 0 or not self._playing.get(channel):
            # Hard cut or nothing playing — just play immediately
            sound = self._mixer.Sound(str(path))
            self.channels[channel].play(sound)
            self._playing[channel] = True
            self._current_track[channel] = path
            self._current_sound[channel] = sound
            self._volumes[channel] = 1.0
            self._apply_volumes()
            return

        # Move old track to fade channel so both overlap
        old_sound = self._current_sound.get(channel)
        if old_sound is not None:
            old_vol = self._volumes[channel]
            self._fade_channel.play(old_sound)
            self._fade_channel.set_volume(old_vol)

        # Start the new track on the main channel at volume 0
        sound = self._mixer.Sound(str(path))
        self.channels[channel].play(sound)
        self._playing[channel] = True
        self._current_track[channel] = path
        self._current_sound[channel] = sound
        self._volumes[channel] = 0.0
        self._apply_volumes()

        self.fade_state = {
            "direction": "crossfade",
            "channel": channel,
            "duration_ms": duration_ms,
            "elapsed_ms": 0,
        }
        self._start_fade_timer()

    def update_fade(self, elapsed_ms: int) -> None:
        """Advance the fade envelope by elapsed_ms and update volume."""
        if self.fade_state is None:
            return

        channel = self.fade_state["channel"]
        duration = self.fade_state["duration_ms"]
        self.fade_state["elapsed_ms"] += elapsed_ms
        progress = min(1.0, self.fade_state["elapsed_ms"] / max(1, duration))

        if self.fade_state["direction"] == "crossfade":
            # Fade in new track on main channel, fade out old on fade channel
            self._volumes[channel] = progress
            self._fade_channel.set_volume(1.0 - progress)
            self._apply_volumes()
            if progress >= 1.0:
                self._fade_channel.stop()
                self.fade_state = None
        elif self.fade_state["direction"] == "in":
            self._volumes[channel] = progress
            self._apply_volumes()
            if progress >= 1.0:
                self.fade_state = None
        elif self.fade_state["direction"] == "out":
            self._volumes[channel] = 1.0 - progress
            self._apply_volumes()
            if progress >= 1.0:
                self.channels[channel].stop()
                self._playing[channel] = False
                self._current_track[channel] = None
                self._current_sound[channel] = None
                self._volumes[channel] = 0.0
                self.fade_state = None

    # -- fade timer (auto-advance in production) --------------------------------

    def _start_fade_timer(self) -> None:
        """Start a background timer to auto-advance fades."""
        if self._fade_timer is not None:
            self._fade_timer.cancel()
        self._fade_timer_ts = time.monotonic()
        self._tick_fade()

    def _tick_fade(self) -> None:
        """One tick of the fade timer."""
        if self.fade_state is None or self._shutdown:
            self._fade_timer = None
            return
        now = time.monotonic()
        elapsed_ms = int((now - self._fade_timer_ts) * 1000)
        self._fade_timer_ts = now
        if elapsed_ms > 0:
            self.update_fade(elapsed_ms)
        if self.fade_state is not None:
            self._fade_timer = threading.Timer(_FADE_TICK_S, self._tick_fade)
            self._fade_timer.daemon = True
            self._fade_timer.start()

    # -- ducking notification -----------------------------------------------

    def notify_channel_done(self, channel: str) -> None:
        self._require_channel(channel)
        self._playing[channel] = False
        if channel == "sfx":
            self._ducked = False
            self._apply_volumes()

    # -- snapshot -----------------------------------------------------------

    def snapshot(self, mood=None):
        """Return a frozen MixerSnapshot of the current mixer state."""
        return MixerSnapshot(
            channel_volumes=dict(self._volumes),
            channel_tracks={ch: self._current_track.get(ch) for ch in CHANNEL_NAMES},
            channel_playing={ch: self._playing.get(ch, False) for ch in CHANNEL_NAMES},
            channel_muted={ch: self._muted.get(ch, False) for ch in CHANNEL_NAMES},
            master_volume=self._master_volume,
            ducked=self._ducked,
            mood=mood,
        )

    # -- shutdown -----------------------------------------------------------

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._fade_timer is not None:
            self._fade_timer.cancel()
            self._fade_timer = None
        self.stop_all()
        self._mixer.quit()
