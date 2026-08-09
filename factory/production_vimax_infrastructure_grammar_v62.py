from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any


_INSTALLED = False
_SPACE_RE = re.compile(r"\s+")


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on"}


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip(" ,.;:")


def _infrastructure_classification_corpus(package: Any) -> str:
    """Return only source/editorial story text that can establish the story type.

    Scene ``visual`` fields are intentionally excluded. They are downstream visual suggestions,
    not source evidence, and a single generic suggestion such as ``secure data center`` must not
    relabel an AI-adoption/software story as physical infrastructure.
    """
    return " ".join(
        [
            str(getattr(package, "topic", "")),
            str(getattr(package, "title", "")),
            str(getattr(package, "narration", "")),
            *(str(getattr(scene, "heading", "")) for scene in getattr(package, "scenes", ()) or ()),
            *(str(getattr(scene, "body", "")) for scene in getattr(package, "scenes", ()) or ()),
        ]
    ).casefold()


def is_ai_infrastructure_story_v62(package: Any) -> bool:
    corpus = _infrastructure_classification_corpus(package)
    strong = (
        "ai factory",
        "ai infrastructure",
        "accelerated computing",
        "compute cluster",
        "gpu cluster",
        "data center",
        "datacenter",
        "server rack",
        "compute rack",
        "high-performance ai infrastructure",
    )
    if any(term in corpus for term in strong):
        return True

    # Supporting terms are deliberately conservative. Generic software stories frequently use
    # words such as ``cloud`` or ``compute``; physical infrastructure grammar is allowed only when
    # at least three distinct infrastructure signals are present and one is a concrete facility /
    # hardware anchor.
    supporting = ("cloud", "server", "rack", "fiber", "cooling", "compute", "infrastructure", "facility")
    physical_anchors = ("server", "rack", "fiber", "cooling", "infrastructure", "facility")
    supporting_hits = {term for term in supporting if term in corpus}
    return len(supporting_hits) >= 3 and any(term in corpus for term in physical_anchors)


_BEAT_DIRECTIONS: tuple[tuple[str, ...], ...] = (
    (
        "wide documentary exterior of a large modern data-center campus under active construction, service vehicles and crews moving equipment near the entrance, no signs or logos",
        "loading-bay view where two technicians wheel a tall unbranded server cabinet from a freight area toward a brightly lit data hall, realistic industrial scale",
        "wide interior aisle of a newly built high-density compute facility while technicians install rack cabinets and overhead cable trays, no readable labels",
        "low-angle infrastructure view of rows of new compute racks, power busways and cooling ducts being commissioned by a small technical crew",
    ),
    (
        "close documentary view of a technician sliding a dense accelerator server tray into an unbranded rack while another engineer routes fiber and power beside it",
        "medium view of an engineer connecting bundles of fiber to network switches above high-density compute nodes with changing unlabelled status lights",
        "technical close view of liquid-cooling manifolds, power distribution and dense compute servers while a technician checks physical connections",
        "wide data-hall view showing several high-density accelerator racks operating together while two engineers validate the installation from the aisle",
    ),
    (
        "research engineers running a physical AI experiment beside a glass-walled compute room, one workstation in foreground and active server racks visible behind them",
        "two researchers comparing results at an unbranded test station directly connected by visible cabling to a nearby high-density compute rack",
        "wide university-style engineering lab connected to a compact compute cluster, researchers moving between experiment hardware and the server area",
        "close hands-on research scene where an engineer connects experimental hardware to a local compute node while a colleague observes the physical output",
    ),
    (
        "very wide data-hall aisle revealing multiple rows of active compute racks, overhead cooling and power infrastructure, with one technician walking through for scale",
        "high-angle view across a dense compute facility showing repeated rack rows, cable trays and cooling infrastructure operating as one large installation",
        "industrial infrastructure scene with technicians commissioning additional server cabinets beside already active compute rows, clearly showing expansion in capacity",
        "wide exterior-to-interior loading scene where new rack cabinets and cooling equipment enter an operating data-center facility, no branding or readable signage",
    ),
    (
        "small group of engineers and generic officials walking through an active data hall while a technician explains a high-density compute rack, candid documentary framing",
        "three engineers collaborating beside an open server rack, one pointing to a physical fiber connection while the others inspect the accelerator hardware",
        "operations-floor scene where infrastructure engineers and visiting technical leaders observe a live rack commissioning procedure, no posed handshake or signage",
        "candid team scene around a compute aisle as one technician installs hardware and two colleagues verify power, network and cooling connections",
    ),
    (
        "final hero wide shot of a large active data-center hall with many illuminated compute racks and technicians moving through the aisle, realistic industrial lighting",
        "dusk exterior of a modern compute facility with cooling equipment and service activity visible through the entrance, no signs, logos or fantasy elements",
        "close-to-wide reveal from one active accelerator rack to several rows of operating infrastructure while an engineer walks into the aisle",
        "high-angle closing view of a large unbranded compute facility combining racks, cooling, power and a small engineering crew in one coherent scene",
    ),
)

_MOTIONS = (
    "Slow forward track while technicians and equipment move naturally through the physical workflow.",
    "Controlled lateral track following the primary technician while hands, cables and equipment visibly change position.",
    "Gentle pull back revealing the scale of the compute facility while people and status lights continue moving.",
    "Slow diagonal dolly through the infrastructure while the installation or commissioning action visibly progresses.",
)


def _direction_for(index: int, beat: int) -> str:
    options = _BEAT_DIRECTIONS[min(max(beat, 0), len(_BEAT_DIRECTIONS) - 1)]
    return options[index % len(options)]


def apply_ai_infrastructure_grammar_v62(plan: Any, package: Any) -> Any:
    """Give physical AI-infrastructure stories a facility-first editorial grammar.

    The exact factual claim remains in the director prompt for review/audit, but it is deliberately
    excluded from the executable visual direction.  This prevents ambiguous company names such as
    "Firebird" from turning into birds, superheroes, logos, or other literal brand imagery.
    """
    if not str(getattr(plan, "prompt_version", "")).startswith("vimax-script2video@"):
        return plan
    if not is_ai_infrastructure_story_v62(package):
        return plan
    package_scenes = list(getattr(package, "scenes", ()) or ())
    if not package_scenes:
        return plan

    from .production_vimax_temporal_video_v55 import _repair_motion_prompt
    from .production_vimax_visual_authority_v52 import _camera_hint

    updated = []
    shot_count = len(plan.scenes)
    for index, scene in enumerate(plan.scenes):
        beat = min(len(package_scenes) - 1, index * len(package_scenes) // max(1, shot_count))
        package_scene = package_scenes[beat]
        claim = _clean(getattr(package_scene, "body", "")) or _clean(getattr(package_scene, "heading", ""))
        direction = _direction_for(index, beat)
        treatment = _camera_hint(str(scene.image_prompt))
        prompt = (
            f"[VIMAX_SHOT_INDEX={index}] "
            f"Factual technology documentary shot synchronized to this exact spoken sentence: {claim}. "
            f"Supporting source-grounded visual direction: {direction}. "
            f"Shot treatment: {treatment}. "
            f"ViMax first frame: {direction}."
        )
        motion = _repair_motion_prompt(_MOTIONS[index % len(_MOTIONS)], index)
        updated.append(
            replace(
                scene,
                image_prompt=prompt,
                motion_prompt=motion,
                continuity_anchor=(
                    "same realistic unbranded AI-infrastructure facility; graphite server racks, "
                    "visible fiber/power/cooling, natural technical crew, consistent cool-neutral practical lighting"
                ),
            )
        )
    return replace(plan, scenes=tuple(updated))


def install_production_vimax_infrastructure_grammar_v62() -> None:
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import production_vimax_visual_authority_v52 as authority_v52

    current = authority_v52._enrich_from_vimax_artifact
    if getattr(current, "_agf_v62", False):
        _INSTALLED = True
        return

    def enrich_v62(plan: Any, package: Any) -> Any:
        return apply_ai_infrastructure_grammar_v62(current(plan, package), package)

    enrich_v62._agf_v62 = True  # type: ignore[attr-defined]
    authority_v52._enrich_from_vimax_artifact = enrich_v62
    _INSTALLED = True
