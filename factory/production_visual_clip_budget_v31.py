from __future__ import annotations

from .visual_storyboard_v30 import clean


_INSTALLED = False


def compact_negative_clip_safe_v31() -> str:
    """Keep text/anatomy/equipment defects inside a conservative SDXL CLIP budget.

    People are not globally mandatory because valid documentary shots may be equipment-only.
    """
    return clean(
        "readable text, pseudo-text, gibberish, logo, watermark, printed label, engraved markings, "
        "screen, collage, empty architecture, vacant scene, humanoid robot, duplicate people, "
        "malformed anatomy, extra limbs, bad hands, warped equipment, broken gear, blurry face, "
        "corridor, blocks, orb"
    )


def install_production_visual_clip_budget_v31() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_visual_subject_authority_v31 as authority

    authority._compact_negative = compact_negative_clip_safe_v31
    _INSTALLED = True
