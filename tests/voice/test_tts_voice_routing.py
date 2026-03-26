"""Tests for TTS voice routing — story 4-6.

Covers all ACs: narrator routing, known NPC routing, default fallback,
model typing, speed preservation, source tracking, speaker detection,
genre pack loading, and empty config handling.
"""

from __future__ import annotations

import pytest

from sidequest_daemon.voice.tts_routing import (
    AssignmentSource,
    Speaker,
    TtsModel,
    TtsVoiceRouter,
    VoiceAssignment,
    VoicePreset,
    identify_speaker,
)


# ---------------------------------------------------------------------------
# AC: Model typing — TtsModel::Kokoro and TtsModel::Piper, no raw strings
# ---------------------------------------------------------------------------


class TestTtsModel:
    def test_kokoro_variant_exists(self) -> None:
        assert TtsModel.Kokoro.value == "kokoro"

    def test_piper_variant_exists(self) -> None:
        assert TtsModel.Piper.value == "piper"

    def test_no_raw_string_construction(self) -> None:
        """TtsModel must be constructed from the enum, not arbitrary strings."""
        with pytest.raises(ValueError):
            TtsModel("invalid_engine")

    def test_kokoro_is_not_piper(self) -> None:
        assert TtsModel.Kokoro != TtsModel.Piper


# ---------------------------------------------------------------------------
# AC: Source tracking — AssignmentSource indicates explicit vs default
# ---------------------------------------------------------------------------


class TestAssignmentSource:
    def test_genre_pack_explicit_exists(self) -> None:
        assert AssignmentSource.GenrePackExplicit.value == "genre_pack_explicit"

    def test_genre_pack_default_exists(self) -> None:
        assert AssignmentSource.GenrePackDefault.value == "genre_pack_default"

    def test_session_override_exists(self) -> None:
        assert AssignmentSource.SessionOverride.value == "session_override"


# ---------------------------------------------------------------------------
# VoicePreset — model, voice_id, and speed
# ---------------------------------------------------------------------------


class TestVoicePreset:
    def test_preset_has_model(self) -> None:
        preset = VoicePreset(model=TtsModel.Kokoro, voice_id="en_male_deep", speed=0.95)
        assert preset.model == TtsModel.Kokoro

    def test_preset_has_voice_id(self) -> None:
        preset = VoicePreset(model=TtsModel.Piper, voice_id="en_US-lessac-medium", speed=1.0)
        assert preset.voice_id == "en_US-lessac-medium"

    def test_preset_has_speed(self) -> None:
        """AC: Speed preserved — voice speed from genre pack config passed through."""
        preset = VoicePreset(model=TtsModel.Kokoro, voice_id="en_male_deep", speed=0.95)
        assert preset.speed == pytest.approx(0.95)

    def test_preset_rejects_negative_speed(self) -> None:
        """Speed must be positive — ValueError, not TypeError from missing init."""
        with pytest.raises(ValueError, match="speed"):
            VoicePreset(model=TtsModel.Kokoro, voice_id="test", speed=-1.0)

    def test_preset_rejects_zero_speed(self) -> None:
        """Speed must be positive — ValueError, not TypeError from missing init."""
        with pytest.raises(ValueError, match="speed"):
            VoicePreset(model=TtsModel.Kokoro, voice_id="test", speed=0.0)


# ---------------------------------------------------------------------------
# VoiceAssignment — routing result with source tracking
# ---------------------------------------------------------------------------


class TestVoiceAssignment:
    def test_assignment_has_character_id(self) -> None:
        preset = VoicePreset(model=TtsModel.Kokoro, voice_id="test", speed=1.0)
        assignment = VoiceAssignment(
            character_id="grimjaw",
            preset=preset,
            source=AssignmentSource.GenrePackExplicit,
        )
        assert assignment.character_id == "grimjaw"

    def test_assignment_has_preset(self) -> None:
        preset = VoicePreset(model=TtsModel.Piper, voice_id="en_US-lessac-medium", speed=1.0)
        assignment = VoiceAssignment(
            character_id="narrator",
            preset=preset,
            source=AssignmentSource.GenrePackExplicit,
        )
        assert assignment.preset.model == TtsModel.Piper

    def test_assignment_has_source(self) -> None:
        preset = VoicePreset(model=TtsModel.Kokoro, voice_id="test", speed=1.0)
        assignment = VoiceAssignment(
            character_id="unknown_npc",
            preset=preset,
            source=AssignmentSource.GenrePackDefault,
        )
        assert assignment.source == AssignmentSource.GenrePackDefault


# ---------------------------------------------------------------------------
# Speaker identification
# ---------------------------------------------------------------------------


class TestSpeaker:
    def test_narrator_variant(self) -> None:
        """Speaker.Narrator must be constructable."""
        speaker = Speaker.narrator()
        assert speaker.is_narrator is True

    def test_character_variant(self) -> None:
        """Speaker.Character('Grimjaw') must carry the character ID."""
        speaker = Speaker.character("Grimjaw")
        assert speaker.is_narrator is False
        assert speaker.character_id == "Grimjaw"

    def test_narrator_has_no_character_id(self) -> None:
        speaker = Speaker.narrator()
        assert speaker.character_id is None


class TestIdentifySpeaker:
    """AC: Speaker detection — 'Grimjaw says: hello' identifies speaker as Grimjaw."""

    def test_character_says_pattern(self) -> None:
        speaker = identify_speaker("Grimjaw says: Stand back!", ["Grimjaw", "Whisper"])
        assert speaker.is_narrator is False
        assert speaker.character_id == "Grimjaw"

    def test_character_colon_pattern(self) -> None:
        speaker = identify_speaker("Whisper: Follow me quietly.", ["Grimjaw", "Whisper"])
        assert speaker.is_narrator is False
        assert speaker.character_id == "Whisper"

    def test_narrator_when_no_match(self) -> None:
        """Text with no dialogue tag defaults to narrator."""
        speaker = identify_speaker(
            "The wind howled through the canyon.", ["Grimjaw", "Whisper"]
        )
        assert speaker.is_narrator is True

    def test_empty_npc_list_always_narrator(self) -> None:
        speaker = identify_speaker("Grimjaw says: Hello!", [])
        assert speaker.is_narrator is True

    def test_case_sensitive_matching(self) -> None:
        """NPC names must match case in known_npcs list."""
        speaker = identify_speaker("grimjaw says: hello", ["Grimjaw"])
        # Exact case match required — lowercase 'grimjaw' should not match 'Grimjaw'
        assert speaker.is_narrator is True

    def test_partial_name_no_false_positive(self) -> None:
        """'Grim' should not match 'Grimjaw'."""
        speaker = identify_speaker("Grim says: hello", ["Grimjaw"])
        assert speaker.is_narrator is True


# ---------------------------------------------------------------------------
# TtsVoiceRouter — core routing logic
# ---------------------------------------------------------------------------


def _sample_media_config() -> dict:
    """Genre pack media config fixture matching the story context YAML."""
    return {
        "voice_presets": {
            "narrator": {"model": "kokoro", "voice": "en_male_deep", "speed": 0.95},
            "default_npc": {
                "model": "piper",
                "voice": "en_US-lessac-medium",
                "speed": 1.0,
            },
            "characters": {
                "grimjaw": {"model": "kokoro", "voice": "en_male_gruff", "speed": 0.9},
                "whisper": {
                    "model": "piper",
                    "voice": "en_GB-alba-medium",
                    "speed": 1.1,
                },
            },
        },
    }


class TestTtsVoiceRouterFromGenrePack:
    """AC: Genre pack load — VoiceRouter.from_genre_pack() parses media config."""

    def test_constructs_from_media_config(self) -> None:
        config = _sample_media_config()
        router = TtsVoiceRouter.from_genre_pack(config)
        assert router is not None

    def test_narrator_preset_loaded(self) -> None:
        config = _sample_media_config()
        router = TtsVoiceRouter.from_genre_pack(config)
        narrator = router.route(Speaker.narrator())
        assert narrator.preset.model == TtsModel.Kokoro
        assert narrator.preset.voice_id == "en_male_deep"

    def test_character_presets_loaded(self) -> None:
        config = _sample_media_config()
        router = TtsVoiceRouter.from_genre_pack(config)
        assignment = router.route(Speaker.character("grimjaw"))
        assert assignment.preset.voice_id == "en_male_gruff"


class TestTtsVoiceRouterNarratorRouting:
    """AC: Narrator routing — narrator text routes to narrator voice preset."""

    def test_narrator_returns_narrator_preset(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.narrator())
        assert assignment.character_id == "narrator"
        assert assignment.preset.model == TtsModel.Kokoro

    def test_narrator_source_is_explicit(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.narrator())
        assert assignment.source == AssignmentSource.GenrePackExplicit

    def test_narrator_speed_preserved(self) -> None:
        """AC: Speed preserved."""
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.narrator())
        assert assignment.preset.speed == pytest.approx(0.95)


class TestTtsVoiceRouterKnownNpc:
    """AC: Known NPC routing — NPC with explicit preset gets that preset."""

    def test_known_npc_gets_explicit_preset(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("grimjaw"))
        assert assignment.preset.model == TtsModel.Kokoro
        assert assignment.preset.voice_id == "en_male_gruff"

    def test_known_npc_source_is_explicit(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("grimjaw"))
        assert assignment.source == AssignmentSource.GenrePackExplicit

    def test_known_npc_speed_preserved(self) -> None:
        """AC: Speed preserved for character-specific preset."""
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("grimjaw"))
        assert assignment.preset.speed == pytest.approx(0.9)

    def test_second_character_preset(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("whisper"))
        assert assignment.preset.model == TtsModel.Piper
        assert assignment.preset.voice_id == "en_GB-alba-medium"
        assert assignment.preset.speed == pytest.approx(1.1)


class TestTtsVoiceRouterDefaultFallback:
    """AC: Default fallback — unknown NPC falls back to default_npc preset."""

    def test_unknown_npc_gets_default_preset(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("unknown_goblin"))
        assert assignment.preset.model == TtsModel.Piper
        assert assignment.preset.voice_id == "en_US-lessac-medium"

    def test_unknown_npc_source_is_default(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("unknown_goblin"))
        assert assignment.source == AssignmentSource.GenrePackDefault

    def test_unknown_npc_preserves_character_id(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("unknown_goblin"))
        assert assignment.character_id == "unknown_goblin"

    def test_unknown_npc_speed_from_default(self) -> None:
        router = TtsVoiceRouter.from_genre_pack(_sample_media_config())
        assignment = router.route(Speaker.character("unknown_goblin"))
        assert assignment.preset.speed == pytest.approx(1.0)


class TestTtsVoiceRouterEmptyConfig:
    """AC: Empty config — missing characters section defaults all NPCs to default_npc."""

    def test_no_characters_section(self) -> None:
        config = {
            "voice_presets": {
                "narrator": {"model": "kokoro", "voice": "en_male_deep", "speed": 0.95},
                "default_npc": {
                    "model": "piper",
                    "voice": "en_US-lessac-medium",
                    "speed": 1.0,
                },
            },
        }
        router = TtsVoiceRouter.from_genre_pack(config)
        assignment = router.route(Speaker.character("any_npc"))
        assert assignment.preset.model == TtsModel.Piper
        assert assignment.source == AssignmentSource.GenrePackDefault

    def test_empty_characters_section(self) -> None:
        config = {
            "voice_presets": {
                "narrator": {"model": "kokoro", "voice": "en_male_deep", "speed": 0.95},
                "default_npc": {
                    "model": "piper",
                    "voice": "en_US-lessac-medium",
                    "speed": 1.0,
                },
                "characters": {},
            },
        }
        router = TtsVoiceRouter.from_genre_pack(config)
        assignment = router.route(Speaker.character("any_npc"))
        assert assignment.source == AssignmentSource.GenrePackDefault


# ---------------------------------------------------------------------------
# Python lang-review rule enforcement
# ---------------------------------------------------------------------------


class TestLangReviewRules:
    """Rule-enforcement tests from python.md lang-review checklist."""

    def test_tts_model_rejects_invalid_value(self) -> None:
        """Rule #8 (unsafe deserialization): reject invalid model strings."""
        with pytest.raises(ValueError):
            TtsModel("whisperx")

    def test_voice_preset_no_mutable_default(self) -> None:
        """Rule #2 (mutable defaults): VoicePreset must not share mutable state."""
        p1 = VoicePreset(model=TtsModel.Kokoro, voice_id="test", speed=1.0)
        p2 = VoicePreset(model=TtsModel.Kokoro, voice_id="test", speed=1.0)
        # Must actually construct — verify they have independent state
        assert p1 is not p2
        assert p1.voice_id == "test"
        assert p2.voice_id == "test"

    def test_identify_speaker_type_annotations(self) -> None:
        """Rule #3 (type annotations): identify_speaker has proper return type."""
        import inspect

        sig = inspect.signature(identify_speaker)
        assert sig.return_annotation is not inspect.Parameter.empty

    def test_from_genre_pack_type_annotations(self) -> None:
        """Rule #3: from_genre_pack has proper parameter annotation."""
        import inspect

        sig = inspect.signature(TtsVoiceRouter.from_genre_pack)
        params = list(sig.parameters.values())
        # First param after cls/self should be annotated
        config_param = params[0] if params[0].name != "cls" else params[1]
        assert config_param.annotation is not inspect.Parameter.empty
