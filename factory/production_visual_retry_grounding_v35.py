from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import production_visual_semantic_review_v28 as semantic_v28
from . import production_visual_subject_authority_v31 as subject_v31
from .visual_storyboard_v30 import clean, storyboard_for


_INSTALLED = False
_MAX_REPAIR_WORDS = 18


def _bounded_words(value: str, limit: int = _MAX_REPAIR_WORDS) -> str:
    return " ".join(clean(value).split()[:limit]).strip(" ,.;:")


def grounded_retry_instruction_v35(*, frame: Any, reviewer_reason: str) -> str:
    """Keep immutable storyboard requirements ahead of reviewer-specific repair feedback."""
    required = _bounded_words(f"show {frame.subject}; {frame.action}", 30)
    correction = _bounded_words(subject_v31._physical_repair(reviewer_reason))
    return clean(f"REQUIRED SUBJECT AND ACTION: {required}. REVIEWER CORRECTION: {correction}")


def scene_for_attempt_v35(
    scene: Any,
    *,
    scene_index: int,
    attempt: int,
    repair: str = "",
) -> Any:
    """Rebuild retries from the immutable scene while retaining exact storyboard requirements."""
    base = semantic_v28._base_director_prompt(str(scene.image_prompt))
    base = clean(subject_v31._STORYBOARD_TAIL_RE.sub("", base))
    frame = storyboard_for(base, scene_index)
    suffix = f". V30 STORYBOARD: shot-{scene_index}; {frame.identity}"
    if repair:
        suffix += f". V31 REPAIR: {grounded_retry_instruction_v35(frame=frame, reviewer_reason=repair)}"
    return replace(
        scene,
        image_prompt=base + suffix,
        negative_prompt=subject_v31._compact_negative(),
        seed=(int(scene.seed) + 161803 * max(0, attempt - 1)) & 0x7FFFFFFF,
    )


def install_production_visual_retry_grounding_v35() -> None:
    """Install v35 compatibility, then hand final authority to the v36 Codex scene gate."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_visual_codex_gate_v36 import (
        install_production_visual_codex_gate_v36,
    )

    semantic_v28._scene_for_attempt = scene_for_attempt_v35
    install_production_visual_codex_gate_v36()
    _INSTALLED = True
