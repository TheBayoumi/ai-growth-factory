from __future__ import annotations

import copy
from typing import Any


_INSTALLED = False
_ROLE_PRIORITY = {
    "mechanism": 60,
    "comparison": 50,
    "evidence": 40,
    "implication": 30,
    "cta": 10,
    "hook": 0,
}


def route_visual_modes(raw: dict[str, Any]) -> dict[str, Any]:
    """Select exactly three Wan hero scenes from the director's semantic roles.

    Qwen remains responsible for scene roles, image prompts, motion prompts, negative
    prompts, and continuity anchors. Model routing is deterministic because an exact
    resource count is an execution policy, not a creative-writing task. Scene zero is
    always the hook. The two remaining hero scenes are selected by explanatory value,
    with stable scene-index tie breaking.
    """
    routed = copy.deepcopy(raw)
    scenes = routed.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return routed

    indexed: dict[int, dict[str, Any]] = {}
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        try:
            index = int(scene.get("scene_index"))
        except (TypeError, ValueError):
            continue
        indexed[index] = scene
    if 0 not in indexed:
        return routed

    candidates: list[tuple[int, int]] = []
    for index, scene in indexed.items():
        if index == 0:
            continue
        role = str(scene.get("role") or "").strip().lower()
        priority = _ROLE_PRIORITY.get(role, 0)
        motion_detail = min(9, len(str(scene.get("motion_prompt") or "").split()) // 8)
        image_detail = min(5, len(str(scene.get("image_prompt") or "").split()) // 30)
        candidates.append((priority + motion_detail + image_detail, index))

    selected = {0}
    for _score, index in sorted(candidates, key=lambda item: (-item[0], item[1]))[:2]:
        selected.add(index)

    for index, scene in indexed.items():
        scene["generation_mode"] = "wan_i2v" if index in selected else "image"
    return routed


def install_production_visual_routing() -> None:
    """Install deterministic resource routing before visual-plan validation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import visual_prompt

    original_validate = visual_prompt._validate_and_normalize

    def validate_with_routing(raw, **kwargs):
        return original_validate(route_visual_modes(raw), **kwargs)

    visual_prompt._validate_and_normalize = validate_with_routing
    _INSTALLED = True
