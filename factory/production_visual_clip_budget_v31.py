from __future__ import annotations

from .visual_storyboard_v30 import clean


_INSTALLED = False


def compact_negative_clip_safe_v31() -> str:
    """Keep the full negative contract inside a conservative SDXL CLIP token budget."""
    return clean(
        "text, letters, numbers, logo, watermark, screen, poster, chart, collage, empty room, "
        "empty server aisle, vacant scene, tiny people, robot person, duplicate people, bad anatomy, "
        "extra limbs, bad hands, broken gear, blurry face, corridor, blocks, orb"
    )


def install_production_visual_clip_budget_v31() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_visual_subject_authority_v31 as authority

    authority._compact_negative = compact_negative_clip_safe_v31
    _INSTALLED = True
