from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Sequence


_INSTALLED = False


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def annotate_remotion_story_beats_v56(spec: Any, source_shots: Sequence[Any]) -> Any:
    """Preserve Factory package-scene boundaries for editor-grade Remotion transitions."""
    if len(spec.shots) != len(source_shots):
        raise ValueError("Remotion story-beat annotation changed the shot count")
    annotated = []
    for render_shot, source_shot in zip(spec.shots, source_shots, strict=True):
        package_scene_index = int(getattr(source_shot, "package_scene_index"))
        purpose = " ".join(str(render_shot.purpose or "support_claim").split())
        annotated.append(
            replace(
                render_shot,
                purpose=f"package_scene:{package_scene_index}; {purpose}",
            )
        )
    updated = replace(spec, shots=tuple(annotated))
    updated.validate(require_files=False)
    return updated


def install_production_remotion_editorial_transitions_v56() -> None:
    """Carry semantic beat identity into the Remotion contract when ViMax is enabled."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import production_remotion_renderer_v45 as remotion_v45

    current = remotion_v45.build_remotion_render_spec
    if getattr(current, "_agf_v56", False):
        _INSTALLED = True
        return

    def build_remotion_render_spec_v56(*args: Any, **kwargs: Any) -> Any:
        source_shots = kwargs.get("shots")
        if source_shots is None:
            raise ValueError("v56 Remotion transition policy requires keyword shot arguments")
        spec = current(*args, **kwargs)
        return annotate_remotion_story_beats_v56(spec, list(source_shots))

    build_remotion_render_spec_v56._agf_v56 = True  # type: ignore[attr-defined]
    remotion_v45.build_remotion_render_spec = build_remotion_render_spec_v56
    _INSTALLED = True
