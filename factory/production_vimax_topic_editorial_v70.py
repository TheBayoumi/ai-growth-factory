from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import replace
from typing import Any, Sequence


_INSTALLED = False
_SPACE_RE = re.compile(r"\s+")
_TEXT_CARD_RE = re.compile(
    r"\b(?:shot of the text|text is displayed|text-based information|readable text|"
    r"headline|web page|dashboard|screen content|interface displays)\b",
    re.IGNORECASE,
)
_STATIC_RE = re.compile(
    r"\b(?:static camera|no movement|no camera movement|no significant changes?|"
    r"unchanged composition|remains in the same position)\b",
    re.IGNORECASE,
)

_PROFESSIONAL_SERVICES_SHOTS: tuple[tuple[str, str, str], ...] = (
    ("office_establish", "single continuous documentary photograph of a modern professional-services office during an active workday, advisers moving between desks and meeting rooms, no readable signs or screens", "Slow forward track through the active office while staff continue moving between work areas."),
    ("client_file", "single continuous documentary photograph of one adviser opening a physical client case folder beside an unbranded laptop with abstract unreadable interface shapes", "Gentle push toward the adviser as the case folder is opened and the laptop workflow continues."),
    ("document_review", "single continuous documentary photograph of two professional advisers comparing printed tax or legal documents at one desk beside an unbranded laptop, realistic hands and papers", "Controlled lateral track across the desk while both advisers turn pages and compare the documents."),
    ("client_conversation", "single continuous documentary photograph of one adviser in a private meeting room speaking with one client across a table, neutral professional setting, no readable documents", "Slow two-person tracking move while the adviser and client continue a natural consultation."),
    ("assisted_workflow", "single continuous documentary photograph of one professional using an unbranded laptop beside a structured paper case file, abstract assistant interface with no readable text", "Gentle over-shoulder push as the adviser moves between the paper file and the laptop workflow."),
    ("quality_check", "single continuous documentary photograph of one adviser checking a drafted client document against two printed source pages, laptop nearby with unreadable interface shapes", "Controlled close track following the adviser from the source pages to the draft during verification."),
    ("team_collaboration", "single continuous documentary photograph of three professional colleagues around one table reviewing physical folders and one unbranded laptop, candid working posture", "Slow arc around the table while colleagues point to folders and continue the working discussion."),
    ("research", "single continuous documentary photograph of one legal or tax professional consulting a shelf of reference binders while carrying a client folder, office library environment", "Controlled lateral move following the professional as one reference binder is selected and opened."),
    ("productivity", "single continuous documentary photograph of one adviser completing several client-work steps at an organized desk with folders, notebook and unbranded laptop, no readable text", "Gentle diagonal track while the adviser moves from one completed folder to the next work item."),
    ("peer_review", "single continuous documentary photograph of two colleagues conducting a peer review of one client brief at a desk, one person annotating paper while the other checks a laptop", "Slow push toward the review as the paper annotation and laptop verification continue together."),
    ("coaching", "single continuous documentary photograph of a senior adviser coaching one colleague at a workstation using a printed case example and an unbranded laptop", "Controlled over-shoulder move as the senior adviser points between the case material and workstation."),
    ("training", "single continuous documentary photograph of a small professional-services team in a training room practicing a workflow on laptops with all screens abstract and unreadable", "Slow lateral track across the training table while participants work through the exercise."),
    ("knowledge_forum", "single continuous documentary photograph of six colleagues in a monthly knowledge-sharing roundtable, one person explaining a process using physical folders and blank cards", "Gentle arc around the roundtable while the speaker gestures and colleagues exchange the physical materials."),
    ("governance", "single continuous documentary photograph of two managers reviewing a stack of internal policy folders beside a closed unbranded laptop in a private office, no readable labels", "Slow push toward the policy review while one manager sorts folders and the other checks the workflow."),
    ("secure_access", "single continuous documentary photograph of one employee using a hardware security key to access an unbranded laptop beside a closed client folder, privacy-conscious office setting", "Tight controlled push as the security key is inserted and the employee secures the client folder."),
    ("confidential_records", "single continuous documentary photograph of one professional scanning a client document into a secure office document station and returning the original to a closed folder", "Controlled track from scanner to folder while the document-handling action completes."),
    ("human_correction", "single continuous documentary photograph of one adviser marking corrections on a printed draft while comparing it with an unbranded laptop, focused quality-control workflow", "Slow close track across the corrections while the adviser alternates between paper and laptop."),
    ("client_brief", "single continuous documentary photograph of one adviser assembling a finished client brief from reviewed pages and a closed folder at an organized desk", "Gentle pull back as the adviser collates the pages and closes the completed client brief."),
    ("service_handoff", "single continuous documentary photograph of two professional colleagues handing off a completed client case folder in a meeting area before one colleague joins a client consultation", "Controlled lateral track following the case-folder handoff toward the consultation area."),
    ("office_hero", "single continuous cinematic documentary photograph of a busy modern advisory office with several professionals working in small groups, one coherent workplace and no readable screens", "Slow cinematic pull back across the active office while multiple small work interactions continue naturally."),
)


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip(" ,.;:")


def factual_story_corpus_v70(package: Any) -> str:
    """Use factual editorial copy only; model-authored visual suggestions are not evidence."""
    parts = [
        str(getattr(package, "topic", "")),
        str(getattr(package, "title", "")),
        str(getattr(package, "narration", "")),
    ]
    for scene in getattr(package, "scenes", ()) or ():
        parts.append(str(getattr(scene, "heading", "")))
        parts.append(str(getattr(scene, "body", "")))
    return _clean(" ".join(parts)).casefold()


def is_ai_infrastructure_story_v70(package: Any) -> bool:
    """Delegate infrastructure routing to v62's factual-only classifier."""
    from .production_vimax_infrastructure_grammar_v62 import is_ai_infrastructure_story_v62

    return bool(is_ai_infrastructure_story_v62(package))


def is_professional_services_story_v70(package: Any) -> bool:
    corpus = factual_story_corpus_v70(package)
    strong = (
        "professional services",
        "tax advisory",
        "tax and legal",
        "tax firm",
        "tax firms",
        "legal firm",
        "legal firms",
        "law firm",
        "advisory work",
        "client service",
        "client services",
        "consulting firm",
        "audit firm",
    )
    if any(term in corpus for term in strong):
        return True
    supporting = (
        "client",
        "employee",
        "employees",
        "productivity",
        "weekly",
        "governance",
        "knowledge sharing",
        "advisory",
        "tax",
        "legal",
        "office",
    )
    return sum(term in corpus for term in supporting) >= 4


def story_world_v70(package: Any) -> str:
    if is_ai_infrastructure_story_v70(package):
        return "ai_infrastructure"
    if is_professional_services_story_v70(package):
        return "professional_services"
    return "vimax_native"


def apply_topic_editorial_storyboard_v70(plan: Any, package: Any) -> Any:
    """Adapt ViMax shot slots to the factual story world without changing timing or ordering."""
    if not str(getattr(plan, "prompt_version", "")).startswith("vimax-script2video@"):
        return plan
    if story_world_v70(package) != "professional_services":
        return plan
    if len(plan.scenes) != len(_PROFESSIONAL_SERVICES_SHOTS):
        raise ValueError(
            f"professional-services HITL grammar requires {len(_PROFESSIONAL_SERVICES_SHOTS)} shots; got {len(plan.scenes)}"
        )
    package_scenes = list(getattr(package, "scenes", ()) or ())
    if not package_scenes:
        raise ValueError("professional-services HITL grammar requires factual package scenes")

    from .production_vimax_temporal_video_v55 import _repair_motion_prompt

    updated = []
    for index, (scene, (_family, direction, motion)) in enumerate(
        zip(plan.scenes, _PROFESSIONAL_SERVICES_SHOTS, strict=True)
    ):
        beat = min(len(package_scenes) - 1, index * len(package_scenes) // len(plan.scenes))
        package_scene = package_scenes[beat]
        claim = _clean(getattr(package_scene, "body", "")) or _clean(getattr(package_scene, "heading", ""))
        prompt = (
            f"[VIMAX_SHOT_INDEX={index}] "
            f"Factual technology documentary shot synchronized to this exact spoken sentence: {claim}. "
            f"Supporting source-grounded visual direction: {direction}. "
            "Shot treatment: one primary physical action, natural documentary framing, single continuous photograph. "
            f"ViMax first frame: {direction}."
        )
        updated.append(
            replace(
                scene,
                image_prompt=prompt,
                motion_prompt=_repair_motion_prompt(motion, index),
                continuity_anchor=(
                    "one coherent modern professional-services workplace; realistic advisers and clients, "
                    "neutral office materials, natural practical lighting, unbranded devices, no readable interface text"
                ),
            )
        )
    return replace(plan, scenes=tuple(updated))


def _direction_signature(value: str) -> str:
    from .production_vimax_visual_authority_v52 import _raw_vimax_direction

    direction = _raw_vimax_direction(str(value)).casefold()
    direction = re.sub(r"\b(?:single continuous|documentary|photograph|one|the|a|an)\b", " ", direction)
    return _clean(direction)


def validate_topic_editorial_diversity_v70(scenes: Sequence[Any]) -> dict[str, int]:
    """Validate the actual final scene set rather than a hardcoded infrastructure family table."""
    scene_list = list(scenes)
    if len(scene_list) < 16:
        raise ValueError(f"ViMax production editorial plan requires at least 16 shots; got {len(scene_list)}")
    signatures = [_direction_signature(str(scene.image_prompt)) for scene in scene_list]
    if any(not value for value in signatures):
        raise ValueError("ViMax production editorial plan contains an empty visual direction")
    counts = Counter(signatures)
    repeated = {key: count for key, count in counts.items() if count > 2}
    if repeated:
        raise ValueError("ViMax production editorial plan repeats a visual direction more than twice: " + str(repeated))
    required_unique = max(8, len(scene_list) // 2)
    if len(counts) < required_unique:
        raise ValueError(
            f"ViMax production editorial plan has only {len(counts)} distinct directions; requires {required_unique}"
        )
    if len(set(signatures[:4])) != 4:
        raise ValueError("ViMax opening four shots must be visually distinct")
    text_cards = [index for index, scene in enumerate(scene_list) if _TEXT_CARD_RE.search(str(scene.image_prompt))]
    if len(text_cards) > 2:
        raise ValueError(f"ViMax production editorial plan contains too many text/interface shots: {text_cards}")
    static = [index for index, scene in enumerate(scene_list) if _STATIC_RE.search(str(scene.motion_prompt))]
    if len(static) > 2:
        raise ValueError(f"ViMax production editorial plan contains too many static shots: {static}")
    return dict(counts)


def install_production_vimax_topic_editorial_v70() -> None:
    """Install a professional-services film grammar after the existing factual story router."""
    global _INSTALLED
    if _INSTALLED or os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() not in {"1", "true", "yes", "on"}:
        return

    from . import production_visual_convergence_v41 as convergence_v41
    from . import production_vimax_visual_authority_v52 as authority_v52

    current = authority_v52._enrich_from_vimax_artifact
    if not getattr(current, "_agf_v70", False):
        def enrich_v70(plan: Any, package: Any) -> Any:
            return apply_topic_editorial_storyboard_v70(current(plan, package), package)

        enrich_v70._agf_v70 = True  # type: ignore[attr-defined]
        authority_v52._enrich_from_vimax_artifact = enrich_v70

    convergence_v41.validate_editorial_contract_diversity_v41 = validate_topic_editorial_diversity_v70
    _INSTALLED = True
