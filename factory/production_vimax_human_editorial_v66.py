from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import replace
from typing import Any, Sequence

from .feeds import SourceItem
from .models import VideoPackage


_INSTALLED = False
_SPACE_RE = re.compile(r"\s+")
_INTERNAL_PROVENANCE_PATTERNS = (
    re.compile(r"\bseparate primary[- ]source context\b", re.IGNORECASE),
    re.compile(r"\beach report is evaluated independently\b", re.IGNORECASE),
    re.compile(r"\bsupports only its attributed claim\b", re.IGNORECASE),
    re.compile(r"\bsource attribution (?:is|was) evaluated\b", re.IGNORECASE),
)
_GENERIC_FILLER_PATTERNS = (
    re.compile(r"\bbroader trend in .* growth\b", re.IGNORECASE),
    re.compile(r"\bwide range of .* applications\b", re.IGNORECASE),
    re.compile(r"\bkey step in expanding .* capabilities\b", re.IGNORECASE),
    re.compile(r"\bsignificant step in global .* development\b", re.IGNORECASE),
)

# Five acts, four shots each. Every direction is intentionally one simple physical action in one
# continuous photograph so SDXL is not encouraged to solve an editing problem inside one frame.
_AI_INFRA_SHOTS: tuple[tuple[str, str, str], ...] = (
    (
        "establish_exterior",
        "single continuous documentary photograph of one modern unbranded data-center building at dusk, one service vehicle approaching the entrance, realistic industrial scale, no inset images, no collage",
        "Slow forward track toward the single facility entrance while the service vehicle continues moving.",
    ),
    (
        "logistics",
        "single continuous documentary photograph inside one industrial loading bay, one technician pushing one tall unbranded server cabinet on a wheeled dolly toward an open data-hall doorway, no collage",
        "Controlled lateral track following the technician and server cabinet through the loading bay.",
    ),
    (
        "commissioning",
        "single continuous documentary photograph of one technician connecting one heavy cable to the rear of an open compute rack in a real data hall, hands and connector clearly visible, no collage",
        "Gentle push toward the connector while the technician completes the physical cable connection.",
    ),
    (
        "hall_scale",
        "single continuous wide documentary photograph down one active high-density compute hall, one technician walking through the aisle for scale, cooling and power infrastructure visible, no collage",
        "Slow pull back through the aisle while the technician continues walking between the rack rows.",
    ),
    (
        "server_hardware",
        "single continuous close documentary photograph of one technician sliding one dense server tray into an open unbranded rack, realistic hands and hardware, no collage",
        "Tight controlled push as the technician slides and seats the server tray into the rack.",
    ),
    (
        "network_fiber",
        "single continuous close documentary photograph of one network engineer connecting one bundle of fiber cables to an unbranded switch above active compute nodes, no readable labels, no collage",
        "Slow lateral move following the engineer's hands across the fiber connections.",
    ),
    (
        "cooling",
        "single continuous close documentary photograph of one technician inspecting a liquid-cooling manifold and coolant hoses beside one compute rack, realistic fittings and tools, no collage",
        "Controlled close track across the cooling manifold while the technician adjusts one fitting.",
    ),
    (
        "power",
        "single continuous documentary photograph of one electrician securing one heavy power connection between an overhead busway and a rack power module, realistic protective equipment, no collage",
        "Slow diagonal move from the busway toward the rack while the electrician secures the connection.",
    ),
    (
        "commissioning_check",
        "single continuous documentary photograph of one infrastructure engineer checking one newly commissioned rack with a small unlabelled handheld meter, active data hall behind, no collage",
        "Gentle push toward the engineer as the meter is moved from one physical connection to the next.",
    ),
    (
        "facility_walkthrough",
        "single continuous documentary photograph of one infrastructure engineer walking beside one visiting technical colleague through an active data hall, candid movement, no handshake, no collage",
        "Controlled tracking move alongside the two technical colleagues as they walk past active racks.",
    ),
    (
        "operations",
        "single continuous documentary photograph of one operations engineer at a physical test bench beside the data hall connecting one unbranded application device to compute infrastructure, no readable interface, no collage",
        "Slow push from the application device toward the engineer while the physical connection is completed.",
    ),
    (
        "maintenance",
        "single continuous documentary photograph of one technician replacing one removable fan module on an operating rack, tool and module clearly visible, realistic anatomy, no collage",
        "Tight track following the fan module as the technician removes and replaces it.",
    ),
    (
        "capacity_delivery",
        "single continuous documentary photograph of one new unbranded server cabinet being rolled beside an already active rack row by one technician, obvious capacity expansion, no collage",
        "Controlled lateral track following the new cabinet toward the active rack row.",
    ),
    (
        "cable_infrastructure",
        "single continuous documentary photograph of one technician routing one fiber bundle into an overhead cable tray above a compute row, ladder stable and environment coherent, no collage",
        "Gentle upward track following the fiber bundle from the rack toward the overhead tray.",
    ),
    (
        "facility_support",
        "single continuous exterior documentary photograph beside one operating data-center building, one technician inspecting a large cooling module and electrical enclosure, no signs, no collage",
        "Slow exterior track past the cooling module while the technician continues the inspection.",
    ),
    (
        "local_engineering",
        "single continuous documentary photograph in one technical training room, two engineers handling one removable compute module on a clean workbench, real equipment only, no robots, no collage",
        "Gentle arc around the workbench while the two engineers inspect and reposition the compute module.",
    ),
    (
        "developer_workflow",
        "single continuous documentary photograph of one developer at an unbranded workstation physically connected to a nearby compact compute rack, abstract unreadable display shapes only, no collage",
        "Slow push from the workstation connection toward the compact serving rack as the developer works.",
    ),
    (
        "application_test",
        "single continuous documentary photograph of one engineer testing one small unbranded edge device on a physical electronics fixture beside compact compute hardware, no robot, no collage",
        "Controlled track from the edge device to the engineer's hands as the physical test progresses.",
    ),
    (
        "team_validation",
        "single continuous documentary photograph of two engineers validating one physical prototype device on a workbench with a compact compute rack behind them, no readable screens, no collage",
        "Gentle push toward the prototype while the engineers manipulate the device and observe the result.",
    ),
    (
        "hero_hall",
        "single continuous cinematic documentary photograph of one large active compute hall with rack rows, cooling and power infrastructure, two engineers walking through the foreground, no collage",
        "Slow cinematic pull back through the active hall while the two engineers continue walking.",
    ),
)


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on"}


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip(" ,.;:")


def consumer_editorial_failures_v66(package: VideoPackage) -> tuple[str, ...]:
    narration = _clean(package.narration)
    failures: list[str] = []
    internal = [pattern.pattern for pattern in _INTERNAL_PROVENANCE_PATTERNS if pattern.search(narration)]
    if internal:
        failures.append("spoken narration contains internal source/provenance process language")
    generic_hits = [pattern.pattern for pattern in _GENERIC_FILLER_PATTERNS if pattern.search(narration)]
    if len(generic_hits) >= 2:
        failures.append(
            f"spoken narration contains {len(generic_hits)} generic filler claims instead of concrete supplied evidence"
        )
    return tuple(failures)


def _repair_scene_actor(package: VideoPackage, sources: Sequence[SourceItem]) -> VideoPackage:
    from .production_editorial_boundary_v65 import _replace_host_actor_with_title_actor

    scenes = []
    changed = False
    for scene in package.scenes:
        if not 0 <= int(scene.source_index) < len(package.source_urls):
            scenes.append(scene)
            continue
        url = package.source_urls[int(scene.source_index)]
        source = next((item for item in sources if item.url == url), None)
        if source is None:
            scenes.append(scene)
            continue
        heading = _replace_host_actor_with_title_actor(scene.heading, source)
        body = _replace_host_actor_with_title_actor(scene.body, source)
        updated = replace(scene, heading=heading, body=body)
        changed = changed or updated != scene
        scenes.append(updated)
    return replace(package, scenes=scenes) if changed else package


def _augment_repair_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nFINAL HUMAN-EDITORIAL SPOKEN COPY CONTRACT (takes precedence over generic filler):\n"
        + "- Write only audience-facing spoken narration. Never describe sourcing, attribution mechanics, report evaluation, or internal evidence handling.\n"
        + "- Forbidden spoken boilerplate includes 'separate primary-source context', 'each report is evaluated independently', and 'supports only its attributed claim'.\n"
        + "- Replace generic phrases such as 'broader trend', 'wide range of applications', and 'key step in expanding capabilities' with concrete facts already present in the supplied source title/summary.\n"
        + "- Do not invent a new fact merely to reach the word target. If the supplied evidence cannot support 130-134 useful spoken words, return skip_reason.\n"
        + "- The first sentence must identify the actual release actor supported by the source, not merely the hosting publisher.\n"
        + "- Scene bodies must use the same supported actor attribution as the narration.\n"
    )


def _install_consumer_copy_gate() -> None:
    from . import local_llm

    current_package_from_raw = local_llm._package_from_raw
    current_repair_prompt = local_llm._repair_prompt

    if not getattr(current_package_from_raw, "_agf_v66", False):
        def package_from_raw_v66(settings: Any, sources: list[SourceItem], raw: dict[str, Any]) -> VideoPackage:
            package = current_package_from_raw(settings, sources, raw)
            package = _repair_scene_actor(package, sources)
            failures = consumer_editorial_failures_v66(package)
            if failures:
                raise local_llm.LocalLLMError("Human editorial copy gate failed: " + "; ".join(failures))
            return package

        package_from_raw_v66._agf_v66 = True  # type: ignore[attr-defined]
        local_llm._package_from_raw = package_from_raw_v66

    if not getattr(current_repair_prompt, "_agf_v66", False):
        def repair_prompt_v66(*args: Any, **kwargs: Any) -> str:
            return _augment_repair_prompt(current_repair_prompt(*args, **kwargs))

        repair_prompt_v66._agf_v66 = True  # type: ignore[attr-defined]
        local_llm._repair_prompt = repair_prompt_v66


def _is_ai_infrastructure(package: Any) -> bool:
    from .production_vimax_infrastructure_grammar_v62 import is_ai_infrastructure_story_v62

    return bool(is_ai_infrastructure_story_v62(package))


def apply_human_editorial_storyboard_v66(plan: Any, package: Any) -> Any:
    if not str(getattr(plan, "prompt_version", "")).startswith("vimax-script2video@"):
        return plan
    if not _is_ai_infrastructure(package):
        return plan
    if len(plan.scenes) != len(_AI_INFRA_SHOTS):
        raise ValueError(f"v66 human-editorial grammar requires {len(_AI_INFRA_SHOTS)} shots; got {len(plan.scenes)}")
    package_scenes = list(getattr(package, "scenes", ()) or ())
    if not package_scenes:
        raise ValueError("v66 human-editorial grammar requires package scenes")

    from .production_vimax_temporal_video_v55 import _repair_motion_prompt

    updated = []
    for index, (scene, (_family, direction, motion)) in enumerate(zip(plan.scenes, _AI_INFRA_SHOTS, strict=True)):
        beat = min(len(package_scenes) - 1, index * len(package_scenes) // len(plan.scenes))
        package_scene = package_scenes[beat]
        claim = _clean(package_scene.body) or _clean(package_scene.heading)
        prompt = (
            f"[VIMAX_SHOT_INDEX={index}] "
            f"Factual technology documentary shot synchronized to this exact spoken sentence: {claim}. "
            f"Supporting source-grounded visual direction: {direction}. "
            "Shot treatment: one primary action, natural documentary framing, single continuous photograph. "
            f"ViMax first frame: {direction}."
        )
        updated.append(
            replace(
                scene,
                image_prompt=prompt,
                motion_prompt=_repair_motion_prompt(motion, index),
                continuity_anchor=(
                    "one coherent unbranded AI-infrastructure documentary world; realistic modern facility materials, "
                    "neutral industrial lighting, consistent practical technical clothing and graphite compute hardware"
                ),
            )
        )
    return replace(plan, scenes=tuple(updated))


def visual_family_counts_v66(scenes: Sequence[Any]) -> dict[str, int]:
    if len(scenes) != len(_AI_INFRA_SHOTS):
        raise ValueError("v66 visual-family validation requires exactly 20 shots")
    families = [family for family, _direction, _motion in _AI_INFRA_SHOTS]
    counts = Counter(families)
    if len(counts) < 16:
        raise ValueError(f"v66 human-editorial plan has insufficient visual-family diversity: {dict(counts)}")
    if len(set(families[:4])) != 4:
        raise ValueError("v66 opening four shots must use four distinct editorial families")
    for index, (_family, direction, _motion) in enumerate(_AI_INFRA_SHOTS):
        lowered = direction.casefold()
        if "single continuous" not in lowered:
            raise ValueError(f"v66 shot {index} is not explicitly one continuous scene")
        if "robot" in lowered and "no robot" not in lowered:
            raise ValueError(f"v66 shot {index} introduces unsupported robotics")
    return dict(counts)


def _install_storyboard_authority() -> None:
    from . import production_visual_convergence_v41 as convergence_v41
    from . import production_vimax_visual_authority_v52 as authority_v52

    current = authority_v52._enrich_from_vimax_artifact
    if not getattr(current, "_agf_v66", False):
        def enrich_v66(plan: Any, package: Any) -> Any:
            return apply_human_editorial_storyboard_v66(current(plan, package), package)

        enrich_v66._agf_v66 = True  # type: ignore[attr-defined]
        authority_v52._enrich_from_vimax_artifact = enrich_v66

    convergence_v41.validate_editorial_contract_diversity_v41 = visual_family_counts_v66


def install_production_vimax_human_editorial_v66() -> None:
    """Install human-editor-approved spoken-copy and keyframe grammar for staged ViMax validation."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return
    _install_consumer_copy_gate()
    _install_storyboard_authority()
    _INSTALLED = True
