from __future__ import annotations

import os
import re
import sys
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
_EDITORIAL_CONTRACT = """
FINAL HUMAN-EDITORIAL SPOKEN COPY CONTRACT (takes precedence over generic filler):
- Write only audience-facing spoken narration. Never describe sourcing, attribution mechanics, report evaluation, or internal evidence handling.
- Forbidden spoken boilerplate includes 'separate primary-source context', 'each report is evaluated independently', and 'supports only its attributed claim'.
- Replace generic phrases such as 'broader trend', 'wide range of applications', and 'key step in expanding capabilities' with concrete facts already present in the supplied source title/summary.
- Do not invent a new fact merely to reach the word target. If the supplied evidence cannot support 130-134 useful spoken words, return skip_reason.
- The first sentence must identify the actual release actor supported by the source, not merely the hosting publisher.
- Scene bodies must use the same supported actor attribution as the narration.
""".strip()

# Five acts, four shots each. Every direction is one simple physical action in one continuous
# photograph, so image generation never has to solve an editing/collage problem inside one frame.
_AI_INFRA_SHOTS: tuple[tuple[str, str, str], ...] = (
    ("establish_exterior", "single continuous documentary photograph of one modern unbranded data-center building at dusk, one service vehicle approaching the entrance, realistic industrial scale", "Slow forward track toward the single facility entrance while the service vehicle continues moving."),
    ("logistics", "single continuous documentary photograph inside one industrial loading bay, one technician pushing one tall unbranded server cabinet on a wheeled dolly toward an open data-hall doorway", "Controlled lateral track following the technician and server cabinet through the loading bay."),
    ("commissioning", "single continuous documentary photograph of one technician connecting one heavy cable to the rear of an open compute rack in a real data hall, hands and connector clearly visible", "Gentle push toward the connector while the technician completes the physical cable connection."),
    ("hall_scale", "single continuous wide documentary photograph down one active high-density compute hall, one technician walking through the aisle for scale, cooling and power infrastructure visible", "Slow pull back through the aisle while the technician continues walking between the rack rows."),
    ("server_hardware", "single continuous close documentary photograph of one technician sliding one dense server tray into an open unbranded rack, realistic hands and hardware", "Tight controlled push as the technician slides and seats the server tray into the rack."),
    ("network_fiber", "single continuous close documentary photograph of one network engineer connecting one bundle of fiber cables to an unbranded switch above active compute nodes", "Slow lateral move following the engineer's hands across the fiber connections."),
    ("cooling", "single continuous close documentary photograph of one technician inspecting a liquid-cooling manifold and coolant hoses beside one compute rack, realistic fittings and tools", "Controlled close track across the cooling manifold while the technician adjusts one fitting."),
    ("power", "single continuous documentary photograph of one electrician securing one heavy power connection between an overhead busway and a rack power module, realistic protective equipment", "Slow diagonal move from the busway toward the rack while the electrician secures the connection."),
    ("commissioning_check", "single continuous documentary photograph of one infrastructure engineer checking one newly commissioned rack with a small unlabelled handheld meter, active data hall behind", "Gentle push toward the engineer as the meter is moved from one physical connection to the next."),
    ("facility_walkthrough", "single continuous documentary photograph of one infrastructure engineer walking beside one visiting technical colleague through an active data hall, candid movement", "Controlled tracking move alongside the two technical colleagues as they walk past active racks."),
    ("operations", "single continuous documentary photograph of one operations engineer at a physical test bench beside the data hall connecting one unbranded application device to compute infrastructure", "Slow push from the application device toward the engineer while the physical connection is completed."),
    ("maintenance", "single continuous documentary photograph of one technician replacing one removable fan module on an operating rack, tool and module clearly visible, realistic anatomy", "Tight track following the fan module as the technician removes and replaces it."),
    ("capacity_delivery", "single continuous documentary photograph of one new unbranded server cabinet being rolled beside an already active rack row by one technician, obvious capacity expansion", "Controlled lateral track following the new cabinet toward the active rack row."),
    ("cable_infrastructure", "single continuous documentary photograph of one technician routing one fiber bundle into an overhead cable tray above a compute row, ladder stable and environment coherent", "Gentle upward track following the fiber bundle from the rack toward the overhead tray."),
    ("facility_support", "single continuous exterior documentary photograph beside one operating data-center building, one technician inspecting a large cooling module and electrical enclosure", "Slow exterior track past the cooling module while the technician continues the inspection."),
    ("local_engineering", "single continuous documentary photograph in one technical training room, two engineers handling one removable compute module on a clean workbench, real technical equipment", "Gentle arc around the workbench while the two engineers inspect and reposition the compute module."),
    ("developer_workflow", "single continuous documentary photograph of one developer at an unbranded workstation physically connected to a nearby compact compute rack, abstract display shapes", "Slow push from the workstation connection toward the compact serving rack as the developer works."),
    ("application_test", "single continuous documentary photograph of one engineer testing one small unbranded edge device on a physical electronics fixture beside compact compute hardware", "Controlled track from the edge device to the engineer's hands as the physical test progresses."),
    ("team_validation", "single continuous documentary photograph of two engineers validating one physical prototype device on a workbench with a compact compute rack behind them", "Gentle push toward the prototype while the engineers manipulate the device and observe the result."),
    ("hero_hall", "single continuous cinematic documentary photograph of one large active compute hall with rack rows, cooling and power infrastructure, two engineers walking through the foreground", "Slow cinematic pull back through the active hall while the two engineers continue walking."),
)


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on"}


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip(" ,.;:")


def consumer_editorial_failures_v66(package: VideoPackage) -> tuple[str, ...]:
    narration = _clean(package.narration)
    failures: list[str] = []
    if any(pattern.search(narration) for pattern in _INTERNAL_PROVENANCE_PATTERNS):
        failures.append("spoken narration contains internal source/provenance process language")
    generic_hits = sum(bool(pattern.search(narration)) for pattern in _GENERIC_FILLER_PATTERNS)
    if generic_hits >= 2:
        failures.append(f"spoken narration contains {generic_hits} generic filler claims instead of concrete supplied evidence")
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
        updated = replace(
            scene,
            heading=_replace_host_actor_with_title_actor(scene.heading, source),
            body=_replace_host_actor_with_title_actor(scene.body, source),
        )
        changed = changed or updated != scene
        scenes.append(updated)
    return replace(package, scenes=scenes) if changed else package


def _augment_repair_prompt(prompt: str) -> str:
    return prompt + "\n\n" + _EDITORIAL_CONTRACT


def _is_package_prompt(prompt: str) -> bool:
    return "SOURCE ENTRIES:" in prompt and (
        "Return one JSON object containing:" in prompt
        or "PREVIOUS JSON:" in prompt
        or "NARRATION REPAIR:" in prompt
    )


def _validate_final_package(package: VideoPackage, sources: Sequence[SourceItem]) -> VideoPackage:
    from . import local_llm

    package = _repair_scene_actor(package, sources)
    failures = consumer_editorial_failures_v66(package)
    if failures:
        raise local_llm.LocalLLMError("Human editorial copy gate failed: " + "; ".join(failures))
    return package


def _install_consumer_copy_gate() -> None:
    from . import local_llm, source_attributed_llm

    current_chat = local_llm._chat
    current_package_from_raw = local_llm._package_from_raw
    current_repair_prompt = local_llm._repair_prompt
    current_source_generate = source_attributed_llm.generate_package

    if not getattr(current_chat, "_agf_v66", False):
        def chat_v66(settings: Any, prompt: str) -> dict[str, Any]:
            if _is_package_prompt(prompt) and _EDITORIAL_CONTRACT not in prompt:
                prompt = prompt + "\n\n" + _EDITORIAL_CONTRACT
            return current_chat(settings, prompt)

        chat_v66._agf_v66 = True  # type: ignore[attr-defined]
        local_llm._chat = chat_v66

    if not getattr(current_package_from_raw, "_agf_v66", False):
        def package_from_raw_v66(settings: Any, sources: list[SourceItem], raw: dict[str, Any]) -> VideoPackage:
            return _validate_final_package(current_package_from_raw(settings, sources, raw), sources)

        package_from_raw_v66._agf_v66 = True  # type: ignore[attr-defined]
        local_llm._package_from_raw = package_from_raw_v66

    if not getattr(current_repair_prompt, "_agf_v66", False):
        def repair_prompt_v66(*args: Any, **kwargs: Any) -> str:
            return _augment_repair_prompt(current_repair_prompt(*args, **kwargs))

        repair_prompt_v66._agf_v66 = True  # type: ignore[attr-defined]
        local_llm._repair_prompt = repair_prompt_v66

    if not getattr(current_source_generate, "_agf_v66", False):
        def source_generate_v66(settings: Any, sources: list[SourceItem], strategy: Any) -> VideoPackage:
            return _validate_final_package(current_source_generate(settings, sources, strategy), sources)

        source_generate_v66._agf_v66 = True  # type: ignore[attr-defined]
        source_attributed_llm.generate_package = source_generate_v66
        canary_module = sys.modules.get("factory.canary")
        if canary_module is not None:
            setattr(canary_module, "generate_package", source_generate_v66)


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
            f"[VIMAX_SHOT_INDEX={index}] Factual technology documentary shot synchronized to this exact spoken sentence: {claim}. "
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
        if "single continuous" not in direction.casefold():
            raise ValueError(f"v66 shot {index} is not explicitly one continuous scene")
        if "robot" in direction.casefold():
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
