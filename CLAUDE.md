# CLAUDE.md — SideQuest Daemon (Python)

Python media services sidecar for SideQuest. Handles image generation
(Flux / Z-Image) and music generation (ACE-Step). Runs alongside the
Python `sidequest-server` (per ADR-082; previously the Rust `sidequest-api`).

## CRITICAL: Personal Project

This is a personal project under the `slabgorb` GitHub account.
- **No Jira integration.** Never create, reference, or interact with Jira tickets.
- **No 1898 org.** Nothing goes to the work GitHub org. Ever.
- All repos live under `github.com/slabgorb/`.

## SideQuest System Overview

Four repos compose the SideQuest stack (Python backend per ADR-082, ported from the Rust prototype 2026-04):
- **sidequest-server** — Python/FastAPI game engine and WebSocket API on port 8765
- **sidequest-ui** — React/TypeScript game client (Vite, port 5173)
- **sidequest-daemon** — Python media services (image gen, audio library playback)
- **sidequest-content** — Genre packs (YAML configs, audio, images, world data)

Orchestrator repo (`orc-quest`, also cloned as `oq-1` / `oq-2`) coordinates sprint tracking, docs, ADRs, and cross-repo scripts.

## Quality Rules

- No stubs, no hacks, no "we'll fix it later" shortcuts
- No skipping tests to save time
- No half-wired features — connect the full pipeline or don't start
- If something needs 5 connections, make 5 connections. Don't ship 3 and call it done.
- **Never say "the right fix is X" and then do Y.** Do X.
- **Never downgrade to a "quick fix" because you think the context is "just a playtest."**
  Every playtest is production tomorrow. Fix it right.

## Development Principles

### No Silent Fallbacks
If something isn't where it should be, fail loudly. Never silently try an alternative
path, config, or default. Silent fallbacks mask configuration problems and lead to
hours of debugging "why isn't this quite right."

### No Stubbing
Don't create stub implementations, placeholder modules, or skeleton code. If a feature
isn't being implemented now, don't leave empty shells for it. Dead code is worse than
no code.

### Don't Reinvent — Wire Up What Exists
Before building anything new, check if the infrastructure already exists in the codebase.
Many systems are fully implemented but not wired into the server or UI. The fix is
integration, not reimplementation.

### Verify Wiring, Not Just Existence
When checking that something works, verify it's actually connected end-to-end. Tests
passing and files existing means nothing if the component isn't imported, the hook isn't
called, or the endpoint isn't hit in production code. Check that new code has non-test
consumers.

### Every Test Suite Needs a Wiring Test
Unit tests prove a component works in isolation. That's not enough. Every set of tests
must include at least one integration test that verifies the component is wired into the
system — imported, called, and reachable from production code paths.

### Backend Language
The server (`sidequest-server`) is Python/FastAPI per ADR-082, ported from a
Rust prototype in 2026-04. The Rust codebase is preserved read-only at
https://github.com/slabgorb/sidequest-api for historical reference; older ADRs
that show Rust code are historical illustration only — see `docs/adr/README.md`
for the translation table. New backend code goes in Python. Media services
(`sidequest-daemon`) remain Python for inference library maturity (Flux /
Z-Image / ACE-Step). Claude calls go through Python subprocesses to the Claude
CLI per ADR-001. (Kokoro TTS was formerly in this list; TTS has been removed
from the system.)

## OTEL Observability Principle

Every backend fix that touches a subsystem MUST add OTEL watcher events so the GM panel
can verify the fix is working. Claude is excellent at "winging it" — writing convincing
narration with zero mechanical backing. The only way to catch this is OTEL logging on
every subsystem decision:

- **Intent classification** — what was the action classified as, and why?
- **Agent routing** — which agent handled the action?
- **State patches** — what changed in game state (HP, location, inventory)?
- **Inventory mutations** — items added/removed, with source
- **NPC registry** — NPCs detected, names assigned, collisions prevented
- **Trope engine** — tick results, keyword matches, activations
- **Encounter engine** — beat selections, metric changes, resolution

The GM panel is the lie detector. If a subsystem isn't emitting OTEL spans, you can't
tell whether it's engaged or whether Claude is just improvising.

**Not needed for:** Cosmetic UI changes (labels, spacing, colors).

## Architecture Decision Index

ADRs live in the orchestrator repo at `orc-quest/docs/adr/`. See
`orc-quest/docs/adr/README.md` for the canonical index. Before designing
or modifying a subsystem, check the relevant ADR:

| Domain | ADRs |
|--------|------|
| Core architecture | 001 (Claude CLI only), 002 (SOUL principles), 005 (background-first), 006 (graceful degradation) |
| Genre packs | 003 (pack architecture), 004 (lazy binding), 072 (system/milieu decomposition — proposed) |
| Prompt engineering | 008 (three-tier taxonomy), 009 (attention-aware zones), 066 (persistent Opus sessions / Full vs Delta tier) |
| Agent system | 010 (intent routing — **superseded by 067**), 011 (JSON patches), 012 (session mgmt), 013 (lazy extraction — superseded by 057), 057 (narrator-crunch separation), 059 (monster manual server-side pregen), 067 (unified narrator agent) |
| Characters | 007 (unified model), 014 (diamonds/coal), 015 (builder FSM), 016 (three-mode chargen), 080 (unified narrative weight) |
| Encounters | 017 (cinematic chase — superseded by 033), 033 (confrontation engine + resource pools), 071 (tactical ASCII grids — proposed) |
| World / NPCs | 018 (trope engine), 019 (cartography), 020 (NPC disposition), 022 (world maturity), 055 (room graph navigation) |
| Progression | 021 (four-track progression), 052 (narrative axis system) |
| Narrative pacing | 024 (dual-track tension), 025 (pacing detection), 050 (image pacing throttle), 051 (two-tier turn counter) |
| Session persistence | 023 (state + recap) |
| Frontend / protocol | 026 (client state mirror), 027 (reactive state messaging), 065 (protocol message decomposition — proposed), 076 (narration protocol collapse post-TTS — proposed), 079 (genre theme system unification) |
| Multiplayer | 028 (perception rewriter), 029 (guest NPC players), 030 (scenario packs), 036 (multiplayer turn coordination), 037 (shared/per-player state split), 053 (scenario system) |
| Transport / IPC | 035 (Unix socket IPC for Python sidecar), 038 (WebSocket transport), 046 (GPU memory budget coordinator), 047 (prompt injection sanitization) |
| Telemetry / Observability | 031 (game watcher semantic telemetry), 058 (Claude subprocess OTEL passthrough), 090 (OTEL dashboard restoration after port) |
| Media | 032 (genre LoRA style training), 034 (portrait identity consistency), 044 (speculative prerender), 048 (lore RAG store), 056 (script tool generators), 070 (MLX image renderer) |
| Dice | 074 (dice resolution protocol — proposed), 075 (3D dice rendering — proposed) |
| Fine-tuning | 069 (scenario fixtures), 073 (local fine-tuned model architecture) |
| Codebase structure | 060 (genre models decomposition), 061 (lore module decomposition), 062 (server lib extraction), 063 (dispatch handler splitting), 064 (game crate domain modules), 068 (magic literal extraction), 088 (ADR frontmatter schema and auto-generated indexes) |
| Project lifecycle | 082 (port back to Python), 085 (tracker hygiene during port) |
| Historical (removed subsystems) | 054 (WebRTC voice chat — files deleted 2026-04), 045 (client audio engine — two-channel post-TTS) |

## Spoiler Protection

- **Fully spoilable:** `mutant_wasteland/flickering_reach` only
- **Fully unspoiled:** Everything else
## Why a separate daemon

This repo exists to keep image-generation library state (Flux / Z-Image
weights, CUDA/MLX contexts) out of the request-handling server process. The
server (`sidequest-server`, Python) calls this daemon over a Unix socket per
ADR-035 — the boundary survived the Rust→Python port (ADR-082) because the
isolation benefits (independent restart, GPU lifecycle, slow warmup) still
apply. Music tracks are generated by the daemon's music pipeline on
operator command (see ADR-095 — Daemon Music Tier via ACE-Step).
Per-track JSON params files in `sidequest-content/genre_packs/<pack>/audio/music/`
are the canonical generation spec. Run `python scripts/generate_music.py
--genre <pack>` to regenerate missing audio.

## Build Commands

```bash
uv sync                  # Install dependencies
uv run pytest            # Run tests
uv run python -m sidequest_daemon  # Run daemon
```

## Architecture

- **Unix socket server** (`/tmp/sidequest-renderer.sock`) — JSON-RPC protocol, routes by `tier` field
- **Image gen**: Flux.1-dev (12 steps) and Flux.1-schnell (4 steps) via diffusers for portraits, landscapes, scenes, tactical maps, text overlays
- **Music**: ACE-Step generation tier. Operator runs `scripts/generate_music.py --genre <pack>`; the script discovers `*_input_params.json` files and dispatches each to the daemon as `tier=music`. Daemon runs ACE-Step → ffmpeg WAV→OGG (libopus 96k) → R2 upload at `genre_packs/<pack>/audio/music/<track>.ogg`. See `sidequest_daemon/media/music_pipeline.py` and ADR-095.
- **Scene interpretation**: Rules-based narration → visual cue extraction (StageCue)
- **Prompt composition**: Tier-specific prefixes, T5 (512 tok) + CLIP (77 tok) budgeting
- **Config**: Reads genre pack paths from `SIDEQUEST_GENRE_PACKS` env var

## Git Workflow

- Branch strategy: gitflow
- Default branch: develop
- Feature branches: `feat/{description}`
- PRs target: develop
