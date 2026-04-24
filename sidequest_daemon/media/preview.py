"""sidequest-promptpreview CLI — print the composed prompt for any target."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sidequest_daemon.media.camera_specs import CameraLoader
from sidequest_daemon.media.catalogs import (
    CharacterCatalog,
    PlaceCatalog,
    StyleCatalog,
)
from sidequest_daemon.media.prompt_composer import PromptComposer
from sidequest_daemon.media.recipe_loader import RecipeLoader
from sidequest_daemon.media.recipes import (
    CameraPreset,
    CatalogMissError,
    RenderTarget,
)

_DAEMON_ROOT = Path(__file__).resolve().parents[2]


def _build_composer(genre: str, world: str) -> PromptComposer:
    packs_root = Path(os.environ.get("SIDEQUEST_GENRE_PACKS", ""))
    if not packs_root.exists():
        raise SystemExit(
            f"SIDEQUEST_GENRE_PACKS does not exist: {packs_root!r}",
        )
    return PromptComposer(
        recipes=RecipeLoader.from_file(_DAEMON_ROOT / "recipes.yaml"),
        cameras=CameraLoader.from_file(_DAEMON_ROOT / "cameras.yaml"),
        characters=CharacterCatalog.load(packs_root, genre=genre, world=world),
        places=PlaceCatalog.load(packs_root, genre=genre, world=world),
        styles=StyleCatalog.load(packs_root, genre=genre, world=world),
    )


def _build_target(args: argparse.Namespace) -> RenderTarget:
    kwargs: dict = {"kind": args.kind, "world": args.world, "genre": args.genre}
    if args.kind == "portrait":
        kwargs["character"] = args.character
        if args.background:
            kwargs["background"] = args.background
        if args.pose_override:
            kwargs["pose_override"] = args.pose_override
        if args.camera:
            kwargs["camera"] = CameraPreset(args.camera)
    elif args.kind == "poi":
        kwargs["place"] = args.place
    elif args.kind == "illustration":
        kwargs["participants"] = [
            p.strip() for p in args.participants.split(",") if p.strip()
        ]
        kwargs["location"] = args.location
        kwargs["action"] = args.action
        kwargs["camera"] = CameraPreset(args.camera)
    return RenderTarget(**kwargs)


def _format_text(target: RenderTarget, result) -> str:
    lines: list[str] = []
    lines.append("== Target ==")
    lines.append(f"kind:         {target.kind}")
    lines.append(f"world:        {target.world}")
    lines.append(f"genre:        {target.genre}")
    lines.append("")
    lines.append("== Composed prompt ==")
    lines.append(result.positive_prompt)
    lines.append("")
    lines.append("== Layer breakdown ==")
    lines.append(f"{'slot':<30} {'source':<40} {'tokens':>7}")
    for layer in result.layers:
        lines.append(
            f"{layer.slot:<30} {layer.source:<40} {layer.estimated_tokens:>7}",
        )
    total = sum(layer.estimated_tokens for layer in result.layers)
    lines.append(f"{'':<30} {'':<40} {'-' * 7}")
    lines.append(f"{'':<30} {'':<40} {total:>7}  (of 512 T5 budget)")
    lines.append("")
    lines.append("== Warnings ==")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"- {w}")
    else:
        lines.append("(none)")
    return "\n".join(lines) + "\n"


def _format_json(result) -> str:
    return json.dumps(result.model_dump(), indent=2) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sidequest-promptpreview")
    subs = p.add_subparsers(dest="kind", required=True)

    def _shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--world", required=True)
        sp.add_argument("--genre", required=True)
        sp.add_argument("--json", action="store_true")

    portrait = subs.add_parser("portrait")
    portrait.add_argument("--character", required=True)
    portrait.add_argument("--background", default=None)
    portrait.add_argument("--pose-override", default=None)
    portrait.add_argument("--camera", default=None)
    _shared(portrait)

    poi = subs.add_parser("poi")
    poi.add_argument("--place", required=True)
    _shared(poi)

    illus = subs.add_parser("illustration")
    illus.add_argument("--participants", required=True, help="comma-separated refs")
    illus.add_argument("--location", required=True)
    illus.add_argument("--action", required=True)
    illus.add_argument("--camera", required=True)
    _shared(illus)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        target = _build_target(args)
        composer = _build_composer(args.genre, args.world)
        result = composer.compose(target)
    except CatalogMissError as e:
        sys.stderr.write(f"catalog miss: {e}\n")
        return 2
    except ValueError as e:
        sys.stderr.write(f"invalid target: {e}\n")
        return 3

    if args.json:
        sys.stdout.write(_format_json(result))
    else:
        sys.stdout.write(_format_text(target, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
