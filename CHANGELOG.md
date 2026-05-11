# Changelog

All notable changes to the SideQuest media daemon (image generation,
ACE-Step music, audio mixing).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-11

### Added
- **ACE-Step music tier (ADR-095)** — daemon-side music generation pipeline
  (`music_pipeline.py`) with ffmpeg post-processing and R2 upload. Per-
  track JSON params live in `sidequest-content`, OGG lives in R2.
- **Per-call span scoping for render→embed** — art-style scoping pinned
  via OTEL span context (45-29).
- README + CLAUDE.md rewrite covering the Z-Image + ACE-Step music tier
  and fidelity-tier swap workflow.

### Fixed
- `opentelemetry.sdk` import moved behind `TYPE_CHECKING` to keep cold
  startup paths clean.
- Added `torchcodec` dependency for torchaudio 2.11 save path.
- ACE-Step field renames: `actual_seeds` → `manual_seeds`,
  `audio_path` → `save_path`.
- Pre-existing daemon lint debt cleaned.

## [1.0.0] - prior

Initial Z-Image MLX renderer + Unix socket IPC baseline (ADR-035,
ADR-070). Not formally tagged at the time; recorded here for continuity.
