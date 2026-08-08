from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any


_INSTALLED = False
_SPACE_RE = re.compile(r"\s+")


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {
        "1", "true", "yes", "on"
    }


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip(" ,.;:")


def classify_editorial_domain_v58(package: Any) -> str:
    corpus = " ".join(
        [
            str(getattr(package, "topic", "")),
            str(getattr(package, "title", "")),
            str(getattr(package, "narration", "")),
            *(str(getattr(scene, "visual", "")) for scene in getattr(package, "scenes", ()) or ()),
        ]
    ).casefold()
    rules = (
        ("gaming", ("gaming", "game ", "games ", "geforce now", "quakecon", "esports", "xbox", "playstation", "steam")),
        ("cybersecurity", ("cybersecurity", "cyber security", "ransomware", "malware", "threat", "vulnerability", "security operations", "zero-day")),
        ("robotics", ("robot", "robotics", "autonomous", "factory automation", "industrial automation", "drone")),
        ("cloud", ("cloud platform", "data center", "datacenter", "hosting", "serverless", "infrastructure platform")),
        ("ai_software", ("artificial intelligence", " ai ", "model", "inference", "llm", "agent", "developer", "software", "api", "hugging face")),
        ("consumer_tech", ("phone", "smartphone", "headset", "wearable", "laptop", "camera", "consumer", "device")),
    )
    padded = f" {corpus} "
    for domain, needles in rules:
        if any(needle in padded for needle in needles):
            return domain
    return "technology"


def _gaming_direction(beat: int, variant: int, claim: str) -> str:
    beats = (
        (
            "close over-shoulder view of a gamer using a controller at a premium cloud-gaming setup, colorful text-free fantasy gameplay filling the display",
            "wide living-room gaming setup with one player actively using a controller, vivid text-free game imagery on the television and natural evening light",
            "close documentary view of a green-accent gaming PC, controller and headset while a player launches a cloud game, all display surfaces free of readable UI",
            "side view of a gamer switching among large text-free illustrated game tiles using a controller, one coherent interface-free gaming scene",
        ),
        (
            "crowded gaming convention hall with attendees actively playing at demo stations, controllers in hand, colorful stage lighting and blank unlettered signage",
            "close candid view of two convention attendees trying a cloud game at a demo station while a crowd moves through the hall behind them",
            "wide gaming-event floor with rows of active demo stations, seated players, moving attendees and energetic colored lighting, no logos or readable banners",
            "tracking-style convention scene following a gamer approaching an occupied demo station while nearby attendees play and react",
        ),
        (
            "gamer actively controlling a naval-combat game with ships and ocean action visible as text-free gameplay, controller hands sharp in the foreground",
            "three adjacent gaming stations in one continuous event space showing visibly different text-free game genres while players actively use controllers",
            "close view of a handheld gaming device streaming vivid text-free gameplay while a second larger display shows the same action in soft background focus",
            "candid player reaction at a demo station as a new game scene loads into colorful text-free gameplay, controller and headset clearly visible",
        ),
        (
            "gamer browsing a growing wall of large illustrated game-art tiles using a controller, cover art only with no titles, labels or readable interface text",
            "cloud-gaming data-center aisle with active green status lights and fiber links while a technician connects a serving node used for live game streaming",
            "player moving from a desktop gaming setup to a handheld device while the same colorful text-free gameplay continues, emphasizing access across devices",
            "wide home gaming room with multiple ready controllers and one active player choosing among varied text-free game artwork on a large display",
        ),
        (
            "close documentary view of a high-end green-accent gaming GPU system rendering colorful gameplay while a player uses a controller beside it",
            "cloud-gaming infrastructure rack with active network lights and cooling airflow, connected to a nearby gaming test station showing text-free live gameplay",
            "gamer playing on a television while a compact streaming client and controller sit in foreground, visually linking home play to cloud infrastructure",
            "technician validating a gaming-streaming node beside a live player test station, green accent lighting and text-free gameplay establishing the gaming ecosystem",
        ),
        (
            "wide gaming-event crowd gathered around active demo stations as players celebrate a successful match, energetic but realistic convention atmosphere",
            "close reaction shot of two gaming enthusiasts celebrating beside a demo station, controllers visible and text-free gameplay glowing behind them",
            "moving convention-floor scene with gamers rotating through demo stations and staff assisting players, blank signage and no readable branding",
            "final hero shot of an active cloud-gaming setup surrounded by engaged players at an event, controller action and colorful gameplay visible without text",
        ),
    )
    options = beats[min(max(beat, 0), len(beats) - 1)]
    return f"{options[variant % len(options)]}; visually support this factual claim without literal text: {claim}"


def _cybersecurity_direction(beat: int, variant: int, claim: str) -> str:
    options = (
        "security analyst inspecting a physical network appliance while abstract threat paths glow on an unreadable operations display",
        "close view of a hardware security key being connected to an isolated workstation during a controlled incident-response procedure",
        "two security engineers tracing network cables between segmented appliances in a realistic operations room with no readable UI",
        "forensic workstation beside removable drives and network hardware while an analyst performs a hands-on investigation",
    )
    return f"{options[(beat + variant) % len(options)]}; visually support this factual claim without literal text: {claim}"


def _robotics_direction(beat: int, variant: int, claim: str) -> str:
    options = (
        "engineer calibrating an articulated robot arm at a guarded test cell while the arm moves through a precise pick-and-place task",
        "mobile autonomous robot navigating a marked industrial test area while a technician observes from a safe distance",
        "close view of sensors and end-effector hardware on an active robotics test bench with visible mechanical motion",
        "wide factory-lab scene where two engineers validate a robotic workflow across one continuous production station",
    )
    return f"{options[(beat + variant) % len(options)]}; visually support this factual claim without literal text: {claim}"


def _cloud_direction(beat: int, variant: int, claim: str) -> str:
    options = (
        "technician connecting fiber and power to a live compute rack with changing status lights in a modern data center",
        "wide aisle of active compute infrastructure with one engineer servicing a node and visible airflow containment",
        "close documentary view of network switches, fiber links and compute nodes during a live infrastructure change",
        "operations engineer moving between a compact application test station and the serving rack that powers it",
    )
    return f"{options[(beat + variant) % len(options)]}; visually support this factual claim without literal text: {claim}"


def _ai_direction(beat: int, variant: int, claim: str) -> str:
    options = (
        "developer testing an AI-powered application beside compact inference compute while another engineer observes the physical output",
        "engineer connecting an application device to a model-serving workstation with abstract unreadable software shapes and changing status lights",
        "two developers validating one AI workflow across an application device and a short compute rack in a realistic lab",
        "close view of a developer operating a model-serving test bench with unbranded hardware and no readable interface content",
    )
    return f"{options[(beat + variant) % len(options)]}; visually support this factual claim without literal text: {claim}"


def _consumer_direction(beat: int, variant: int, claim: str) -> str:
    options = (
        "person actively using the featured class of consumer device in a realistic everyday setting, product action clearly visible and no branding",
        "close hands-on product demonstration showing the device's physical interaction and real-world context without readable screens or labels",
        "wide lifestyle scene showing the consumer device being used naturally rather than posed as a product render",
        "technical close-up of the device hardware during an actual user action with realistic materials and natural lighting",
    )
    return f"{options[(beat + variant) % len(options)]}; visually support this factual claim without literal text: {claim}"


def editorial_direction_v58(domain: str, beat: int, variant: int, claim: str) -> str:
    if domain == "gaming":
        return _gaming_direction(beat, variant, claim)
    if domain == "cybersecurity":
        return _cybersecurity_direction(beat, variant, claim)
    if domain == "robotics":
        return _robotics_direction(beat, variant, claim)
    if domain == "cloud":
        return _cloud_direction(beat, variant, claim)
    if domain == "ai_software":
        return _ai_direction(beat, variant, claim)
    if domain == "consumer_tech":
        return _consumer_direction(beat, variant, claim)
    generic = (
        "hands-on technology demonstration in a realistic environment with one clear physical action",
        "candid operator using the technology in a real workflow with visible cause and effect",
        "close documentary view of the key hardware or physical consequence during active use",
        "wide contextual scene showing people using the technology naturally in its real operating environment",
    )
    return f"{generic[(beat + variant) % len(generic)]}; visually support this factual claim without literal text: {claim}"


def _motion_v58(domain: str, index: int) -> str:
    if domain == "gaming":
        motions = (
            "Slow track toward the active player while controller hands move and the text-free gameplay visibly changes.",
            "Controlled pan following a convention attendee between active gaming stations as players and background crowds continue moving.",
            "Gentle push toward controller action while animated gameplay and practical lighting change continuously behind the player.",
            "Slow lateral track revealing the relationship between the player, gaming device and cloud-gaming environment with continuous human and screen motion.",
        )
    else:
        motions = (
            "Slow dolly toward the primary action while the subject and equipment visibly change state.",
            "Controlled lateral track following the operator through the depicted workflow with continuous physical motion.",
            "Gentle push toward the consequence of the action while environment and subject motion continue throughout the shot.",
            "Slow pull back revealing the complete workflow as the depicted task visibly progresses from first frame to last.",
        )
    return motions[index % len(motions)]


def apply_editorial_visual_grammar_v58(plan: Any, package: Any) -> Any:
    if not str(getattr(plan, "prompt_version", "")).startswith("vimax-script2video@"):
        return plan
    package_scenes = list(getattr(package, "scenes", ()) or ())
    if not package_scenes:
        return plan
    domain = classify_editorial_domain_v58(package)
    from .production_vimax_temporal_video_v55 import _repair_motion_prompt
    from .production_vimax_visual_authority_v52 import _camera_hint

    updated = []
    count = len(plan.scenes)
    for index, scene in enumerate(plan.scenes):
        beat = min(len(package_scenes) - 1, index * len(package_scenes) // max(1, count))
        package_scene = package_scenes[beat]
        claim = _clean(getattr(package_scene, "body", "")) or _clean(getattr(package_scene, "heading", ""))
        direction = editorial_direction_v58(domain, beat, index, claim)
        treatment = _camera_hint(str(scene.image_prompt))
        prompt = (
            f"[VIMAX_SHOT_INDEX={index}] "
            f"Factual technology documentary shot synchronized to this exact spoken sentence: {claim}. "
            f"Supporting source-grounded visual direction: {direction}. "
            f"Shot treatment: {treatment}. "
            f"ViMax first frame: {direction}."
        )
        motion = _repair_motion_prompt(_motion_v58(domain, index), index)
        updated.append(replace(scene, image_prompt=prompt, motion_prompt=motion))
    return replace(plan, scenes=tuple(updated))


def install_production_vimax_editorial_grammar_v58() -> None:
    """Replace generic AI-lab physicalization with domain-aware, filmable editorial B-roll."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import production_vimax_visual_authority_v52 as authority_v52

    current = authority_v52._enrich_from_vimax_artifact
    if getattr(current, "_agf_v58", False):
        _INSTALLED = True
        return

    def enrich_v58(plan: Any, package: Any) -> Any:
        return apply_editorial_visual_grammar_v58(current(plan, package), package)

    enrich_v58._agf_v58 = True  # type: ignore[attr-defined]
    authority_v52._enrich_from_vimax_artifact = enrich_v58
    _INSTALLED = True
