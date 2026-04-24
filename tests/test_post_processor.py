from PIL import Image

from sidequest_daemon.media.camera_specs import PostDirective
from sidequest_daemon.media.post_processor import apply_post, required_render_size


def _make_image(size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, color=(128, 0, 0))
    inner = Image.new("RGB", (size[0] // 2, size[1] // 2), color=(255, 255, 255))
    img.paste(inner, (size[0] // 4, size[1] // 4))
    return img


def test_crop_center_25_percent_preserves_center() -> None:
    src = _make_image((4096, 4096))
    directive = PostDirective(kind="crop", mode="center", percent=0.25)
    out = apply_post(src, directive)
    assert out.size == (1024, 1024)
    assert out.getpixel((512, 512)) == (255, 255, 255)


def test_rotate_inscribed_rect() -> None:
    src = _make_image((1024, 1024))
    directive = PostDirective(kind="rotate", degrees=15.0)
    out = apply_post(src, directive)
    assert out.size[0] < src.size[0]
    assert out.size[1] < src.size[1]
    assert out.getpixel((0, 0)) != (0, 0, 0)


def test_no_post_returns_input_unchanged() -> None:
    src = _make_image((512, 512))
    out = apply_post(src, None)
    assert out is src


def test_crop_25_percent_requires_4x_target_per_axis() -> None:
    target_w, target_h = 1024, 1024
    directive = PostDirective(kind="crop", mode="center", percent=0.25)
    w, h = required_render_size((target_w, target_h), directive)
    assert w == 4096
    assert h == 4096


def test_no_post_returns_target() -> None:
    assert required_render_size((1024, 1024), None) == (1024, 1024)


def test_rotate_requires_diagonal_envelope() -> None:
    directive = PostDirective(kind="rotate", degrees=15.0)
    w, h = required_render_size((1024, 1024), directive)
    assert w >= 1200
    assert h >= 1200
