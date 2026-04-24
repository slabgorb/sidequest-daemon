"""Pillow-based post-processing for camera preset directives."""

from __future__ import annotations

import math

from PIL import Image

from sidequest_daemon.media.camera_specs import PostDirective


def apply_post(img: Image.Image, directive: PostDirective | None) -> Image.Image:
    if directive is None:
        return img
    if directive.kind == "crop":
        return _center_crop(img, directive.percent or 1.0)
    if directive.kind == "rotate":
        return _rotate_inscribed(img, directive.degrees or 0.0)
    raise ValueError(f"unknown post kind: {directive.kind!r}")


def _center_crop(img: Image.Image, percent: float) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * percent))
    new_h = max(1, int(h * percent))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def _rotate_inscribed(img: Image.Image, degrees: float) -> Image.Image:
    rotated = img.rotate(degrees, resample=Image.BICUBIC, expand=False)
    w, h = img.size
    theta = math.radians(abs(degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    new_w = int((w * cos_t - h * sin_t) if w >= h else (h * cos_t - w * sin_t))
    new_h = int(new_w * (h / w)) if w >= h else int(new_w * (w / h))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return rotated.crop((left, top, left + new_w, top + new_h))


def required_render_size(
    target_size: tuple[int, int],
    directive: PostDirective | None,
) -> tuple[int, int]:
    if directive is None:
        return target_size
    tw, th = target_size
    if directive.kind == "crop":
        percent = directive.percent or 1.0
        if percent <= 0:
            raise ValueError("crop percent must be > 0")
        return (int(tw / percent), int(th / percent))
    if directive.kind == "rotate":
        theta = math.radians(abs(directive.degrees or 0.0))
        denom = max(1e-6, math.cos(theta) - math.sin(theta))
        return (int(tw / denom) + 1, int(th / denom) + 1)
    return target_size
