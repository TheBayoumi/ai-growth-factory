from __future__ import annotations

from .visual_storyboard_v30 import StoryboardFrame, clean


_INSTALLED = False


_ATOMIC_CONTROLLED_TEST_FRAMES = (
    StoryboardFrame(
        "controlled_test",
        0,
        "a compact controlled test bench with two identical blank devices and two simple status lamps",
        "one adult evaluator beside the two unlabelled devices",
        "the evaluator starts the same physical cycle on both devices and watches the two lamps",
        "balanced eye-level comparison with both devices large and equally visible",
        "dark neutral bench, white fixtures, green and amber lamps",
    ),
    StoryboardFrame(
        "controlled_test",
        1,
        "a clean repeatability bench with one compact robotic gripper above a straight row of identical blank blocks",
        "one compact robotic gripper and one small overhead camera sensor, all surfaces blank and unmarked",
        "the gripper moves one block from the first position to the second while the camera observes",
        "tight three-quarter view with the gripper and block row filling the foreground",
        "cool blue machinery, warm task light, plain neutral background",
    ),
    StoryboardFrame(
        "controlled_test",
        2,
        "a compact measurement bench with one motor rig and two blank result trays",
        "one adult engineer beside the motor rig and two unlabelled trays",
        "the engineer places one tested part into one tray after the motor cycle finishes",
        "eye-level process view showing the motor rig, one part, and both trays",
        "clean gray bench, blue motor rig, red and green trays without markings",
    ),
)


def compact_negative_atomic_v34() -> str:
    """Keep text/anatomy/equipment defects inside a conservative SDXL CLIP window.

    Do not require people globally: several valid documentary shots are intentionally equipment-only.
    """
    return clean(
        "readable text, pseudo-text, gibberish, logo, watermark, printed label, engraved markings, "
        "screen, collage, empty architecture, vacant scene, humanoid robot, duplicate people, "
        "malformed anatomy, extra limbs, bad hands, warped equipment, broken gear, blurry face, "
        "corridor, blocks, orb"
    )


def physical_repair_atomic_v34(reason: str) -> str:
    lowered = clean(reason).casefold()
    if any(
        term in lowered
        for term in (
            "readable text",
            "pseudo-text",
            "gibberish",
            "letters",
            "numbers",
            "label",
            "marking",
        )
    ):
        return "Use smooth blank unmarked equipment surfaces with no labels glyphs numbers or engraved markings"
    if any(term in lowered for term in ("collage", "split", "grid", "multiple frames", "panel")):
        return "Use one uninterrupted photograph in one continuous environment from one camera"
    if any(term in lowered for term in ("malformed", "distorted", "extra limb", "broken equipment", "impossible")):
        return "Use one primary machine one physical action and mechanically plausible geometry"
    if any(
        term in lowered
        for term in (
            "lacks",
            "missing",
            "does not match",
            "does not depict",
            "required elements",
            "subjects and actions",
            "environment",
        )
    ):
        return "Show only the named machine object and single physical action large and unmistakable in the foreground"
    return "Keep one primary subject one supporting object and one clear physical action in the foreground"


def validate_atomic_controlled_test_registry_v34() -> None:
    """Fail source preflight if controlled-test variants become overloaded again."""
    forbidden = (
        "two adult researchers",
        "independent reviewer",
        "rows of identical test objects",
        "microscope",
        "computer",
        "screen",
        "display",
    )
    for frame in _ATOMIC_CONTROLLED_TEST_FRAMES:
        combined = clean(
            f"{frame.environment} {frame.subject} {frame.action} {frame.camera}"
        ).casefold()
        if any(term in combined for term in forbidden):
            raise ValueError(
                f"Controlled-test storyboard {frame.identity} is not atomic: {combined}"
            )
        if len(clean(frame.subject).split()) > 18:
            raise ValueError(f"Controlled-test subject is overloaded: {frame.identity}")
        if len(clean(frame.action).split()) > 22:
            raise ValueError(f"Controlled-test action is overloaded: {frame.identity}")


def install_production_visual_atomic_storyboard_v34() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_visual_clip_budget_v31 as clip_budget
    from . import production_visual_subject_authority_v31 as subject_authority
    from . import visual_storyboard_v30

    validate_atomic_controlled_test_registry_v34()
    visual_storyboard_v30._REGISTRY["controlled_test"] = _ATOMIC_CONTROLLED_TEST_FRAMES
    clip_budget.compact_negative_clip_safe_v31 = compact_negative_atomic_v34
    subject_authority._compact_negative = compact_negative_atomic_v34
    subject_authority._physical_repair = physical_repair_atomic_v34
    _INSTALLED = True
