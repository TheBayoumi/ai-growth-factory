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


def _install_business_environment_diversity_v45() -> None:
    """Give long business-adoption plans five executable environment families.

    The v41 business frame bank originally contained four environments. A twenty-shot
    ViMax plan therefore repeated each family five times and correctly failed the frozen
    five-family editorial gate. Extend the frame bank with a transcript-compatible field
    service deployment rather than weakening that gate or rewriting factual claims.
    """
    from . import production_visual_convergence_v41 as convergence_v41
    from .visual_storyboard_v30 import StoryboardFrame

    frames = tuple(convergence_v41._FRAME_BANKS["business_adoption"])
    if any(
        frame.variant == 4 and "portable field-service" in frame.environment.casefold()
        for frame in frames
    ):
        return

    field_service_frame = StoryboardFrame(
        "business_adoption",
        4,
        "a portable field-service station with a rugged tablet and compact local AI appliance",
        "one field-service technician beside the open rugged equipment case",
        "the technician runs an enterprise diagnostic workflow on the local appliance and records the result on the tablet",
        "medium documentary view with the tablet, appliance, diagnostic cable, and equipment case visible",
        "overcast daylight, rugged gray equipment, orange case accents, green appliance indicators",
    )
    expanded = (*frames, field_service_frame)
    convergence_v41._DIVERSE_BUSINESS_FRAMES = expanded
    convergence_v41._FRAME_BANKS["business_adoption"] = expanded


def install_production_vimax_planning_v45() -> None:
    """Install ViMax as the planning authority only when explicitly enabled."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import production_editorial_v28, visual_pipeline, visual_prompt
    from .production_package_capacity_v46 import (
        install_production_package_capacity_v46,
    )
    from .vimax_planner import (
        build_vimax_editorial_plan,
        construct_vimax_visual_plan,
        persist_vimax_plan_artifact,
    )

    # ViMax can expand a six-scene package into a much denser editorial shot plan. Align the
    # upstream package capacity contract before any content is generated so the migration path
    # does not waste authoritative candidates on contradictory word-count instructions.
    install_production_package_capacity_v46()
    _install_business_environment_diversity_v45()
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
