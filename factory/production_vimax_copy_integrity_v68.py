from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Callable

from .models import VideoPackage


_INSTALLED = False
_SPACE_RE = re.compile(r"\s+")


def normalize_finished_copy_v68(value: object) -> str:
    """Normalize whitespace without deleting meaningful terminal punctuation."""
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def preserve_finished_copy_v68(
    package: VideoPackage,
    raw: dict[str, Any],
    *,
    base_apply: Callable[[VideoPackage, dict[str, Any]], VideoPackage],
) -> VideoPackage:
    """Run the strict v66 structural parser, then restore punctuation on finished copy only.

    Source URLs, publishers, scene order, source indices and visual directions remain owned by
    the v66 structural boundary. This adapter only restores whitespace-normalized narration and
    scene-body strings from the already validated rewrite payload.
    """
    candidate = base_apply(package, raw)
    narration = normalize_finished_copy_v68(raw.get("narration"))
    scenes_raw = raw.get("scenes")
    if not narration or not isinstance(scenes_raw, list):
        return candidate

    body_by_id: dict[int, str] = {}
    for item in scenes_raw:
        if not isinstance(item, dict):
            continue
        scene_id = item.get("scene_id")
        if isinstance(scene_id, bool) or not isinstance(scene_id, int):
            continue
        body = normalize_finished_copy_v68(item.get("body"))
        if body:
            body_by_id[scene_id] = body

    scenes = [
        replace(scene, body=body_by_id.get(index, scene.body))
        for index, scene in enumerate(candidate.scenes)
    ]
    restored = replace(candidate, narration=narration, scenes=scenes)

    if restored.source_urls != candidate.source_urls:
        raise ValueError("v68 copy integrity changed source URLs")
    if restored.source_publishers != candidate.source_publishers:
        raise ValueError("v68 copy integrity changed source publishers")
    for before, after in zip(candidate.scenes, restored.scenes, strict=True):
        if before.source_index != after.source_index or before.visual != after.visual:
            raise ValueError("v68 copy integrity changed immutable scene evidence metadata")
    return restored


def install_production_vimax_copy_integrity_v68() -> None:
    """Preserve finished editorial punctuation after v66's strict rewrite parser."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_vimax_human_editorial_v66 as v66

    current = v66._apply_focused_editorial_rewrite_v66
    if getattr(current, "_agf_v68", False):
        _INSTALLED = True
        return

    def apply_v68(package: VideoPackage, raw: dict[str, Any]) -> VideoPackage:
        return preserve_finished_copy_v68(package, raw, base_apply=current)

    apply_v68._agf_v68 = True  # type: ignore[attr-defined]
    v66._apply_focused_editorial_rewrite_v66 = apply_v68
    _INSTALLED = True
