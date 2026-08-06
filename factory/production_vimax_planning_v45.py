from __future__ import annotations

import os
import sys


_INSTALLED = False


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_production_vimax_planning_v45() -> None:
    """Install ViMax as the planning authority only when explicitly enabled."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import production_editorial_v28, visual_pipeline, visual_prompt
    from .vimax_planner import (
        build_vimax_editorial_plan,
        construct_vimax_visual_plan,
        persist_vimax_plan_artifact,
    )

    current_persist_visual_plan = visual_pipeline.persist_visual_plan

    def persist_visual_plan_v45(plan, output_path):
        result = current_persist_visual_plan(plan, output_path)
        persist_vimax_plan_artifact(plan, output_path.parent)
        return result

    visual_pipeline.persist_visual_plan = persist_visual_plan_v45
    visual_prompt.construct_visual_plan = construct_vimax_visual_plan
    production_editorial_v28.build_editorial_plan = build_vimax_editorial_plan
    for module_name in ("factory.pipeline", "factory.canary"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "construct_visual_plan", construct_vimax_visual_plan)
    _INSTALLED = True
