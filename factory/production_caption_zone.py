from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw


_INSTALLED = False
_TRANSITION_START_RATIO = 0.60
_MATTE_START_RATIO = 0.68
_MATTE_RGB = (5, 7, 12)


def production_caption_safe_zone(
    image: Image.Image,
    *,
    start_ratio: float = _MATTE_START_RATIO,
) -> tuple[Image.Image, float, float]:
    """Replace the lower third with real negative space before visual review.

    A blur-and-darken treatment can leave an essential object visible beneath the captions,
    which caused v24 scene 4 to fail on all three attempts. Production now preserves every
    pixel above 60%, fades smoothly into a neutral matte between 60% and 68%, and makes the
    full lower 32% subject-free. Captions remain a separate ASS layer and no text is added.
    """
    from . import image_generator

    source = image.convert("RGB")
    width, height = source.size
    matte_start = max(1, min(height - 1, round(height * start_ratio)))
    transition_start = max(0, min(matte_start - 1, round(height * _TRANSITION_START_RATIO)))

    original_zone = source.crop((0, matte_start, width, height))
    before = image_generator._detail_score(original_zone)

    result = source.copy()
    matte = Image.new("RGB", (width, height), _MATTE_RGB)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    transition_height = max(1, matte_start - transition_start)
    for row in range(transition_start, matte_start):
        progress = (row - transition_start) / transition_height
        eased = progress * progress * (3.0 - 2.0 * progress)
        draw.line((0, row, width, row), fill=round(255 * eased))
    draw.rectangle((0, matte_start, width, height), fill=255)
    result = Image.composite(matte, result, mask)

    repaired_zone = result.crop((0, matte_start, width, height))
    expected_matte = Image.new("RGB", repaired_zone.size, _MATTE_RGB)
    mismatch = ImageChops.difference(repaired_zone, expected_matte).getbbox()
    if mismatch is not None:
        raise image_generator.ImageGenerationError(
            "Production caption matte is not uniformly subject-free: "
            f"before={before:.3f}, mismatch_bbox={mismatch}"
        )

    # FIND_EDGES adds synthetic border responses even for a perfectly uniform image.
    # Exact pixel equality above is the stronger invariant, so the repaired detail is zero.
    after = 0.0
    return result, before, after


def install_production_caption_zone() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator

    image_generator._caption_safe_zone = production_caption_safe_zone
    _INSTALLED = True
