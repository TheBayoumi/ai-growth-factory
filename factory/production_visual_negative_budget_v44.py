from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from . import production_visual_convergence_v41 as convergence_v41
from . import production_visual_semantic_grounding_v40 as grounded_v40
from . import production_visual_semantic_review_v28 as semantic_v28
from . import production_visual_subject_authority_v31 as subject_v31


_INSTALLED = False
_MAX_NEGATIVE_WORDS = 36


class VisualNegativeBudgetError(ValueError):
    """Raised before GPU inference when mandatory defect bans cannot fit safely."""


_MANDATORY_BASE = (
    "readable text lettering numbers",
    "logo watermark",
    "collage split frame",
)

_CATEGORY_PRIORITY = {
    "local_inference": (
        "public research hub robotics workshop classroom warehouse",
        "outdoor mountain landscape",
        "spacesuit astronaut armor",
        "weapon gun rifle",
        "bicycle motorcycle",
        "futuristic corridor",
    ),
    "model_release": (
        "public research hub generic partnership meeting",
        "classroom warehouse",
        "outdoor mountain landscape",
        "spacesuit astronaut armor",
        "weapon gun rifle",
        "collage contact sheet",
    ),
    "business_adoption": (
        "public research hub robotics workshop classroom warehouse",
        "factory floor",
        "outdoor mountain landscape",
        "spacesuit astronaut armor",
        "weapon gun rifle",
        "bicycle motorcycle",
    ),
    "source_grounded_ai": (
        "outdoor mountain landscape",
        "spacesuit astronaut armor",
        "weapon gun rifle military",
        "bicycle motorcycle",
        "public research hub robotics workshop classroom warehouse",
        "futuristic corridor",
    ),
    "controlled_test": (
        "factory floor heavy industrial machinery",
        "open machine cabinet",
        "multiple workers",
        "robotic arm vehicle repair",
        "vintage black and white photograph",
    ),
}

_OPTIONAL_QUALITY = (
    "malformed anatomy",
    "extra limbs distorted hands",
    "warped equipment",
    "blurry image low resolution",
    "duplicate objects",
    "empty architecture",
    "poster chart infographic",
)


def _words(value: str) -> list[str]:
    return convergence_v41._words(value)


def _semantic_key(value: str) -> str:
    return convergence_v41._semantic_key(value)


def _contains_phrase(container: str, phrase: str) -> bool:
    return convergence_v41._contains_phrase(container, phrase)


def _unique_items(items: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = grounded_v40.clean(item)
        key = _semantic_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return tuple(result)


def _fit_negative_items(
    required: Iterable[str],
    optional: Iterable[str],
    *,
    limit: int = _MAX_NEGATIVE_WORDS,
) -> str:
    kept: list[str] = []
    count = 0
    for item in _unique_items(required):
        size = len(_words(item))
        if count + size > limit:
            raise VisualNegativeBudgetError(
                f"Mandatory negative contract exceeds {limit} words at: {item}"
            )
        kept.append(item)
        count += size
    for item in _unique_items(optional):
        size = len(_words(item))
        if count + size > limit:
            continue
        kept.append(item)
        count += size
    return ", ".join(kept)


def bounded_negative_prompt_v44(
    contract: grounded_v40.GroundedSceneContractV40,
    *,
    reviewer_feedback: str = "",
) -> str:
    """Preserve corrective bans first and drop generic quality terms before CLIP overflow."""
    feedback = convergence_v41._feedback_negatives_v41(reviewer_feedback)
    category = _CATEGORY_PRIORITY.get(contract.category, ())
    optional_contract = tuple(
        item
        for item in contract.forbidden_substitutions
        if not any(_contains_phrase(required, item) for required in contract.required_phrases)
    )
    negative = _fit_negative_items(
        (*feedback, *_MANDATORY_BASE),
        (*category, *optional_contract, *_OPTIONAL_QUALITY),
    )
    conflicts = [
        phrase
        for phrase in contract.required_phrases
        if _contains_phrase(negative, phrase)
    ]
    if conflicts:
        raise VisualNegativeBudgetError(
            f"Bounded negative prompt contradicts required phrases: {conflicts}"
        )
    return negative


def compile_negative_budget_v44(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = convergence_v41._MAX_WORDS,
) -> Any:
    """Reuse v41 positive authority while replacing its unbounded negative contract."""
    del director_negative_prompt
    compiled = convergence_v41.compile_convergent_prompt_v41(
        director_prompt,
        word_budget=word_budget,
    )
    contract = convergence_v41.grounded_contract_for_v41(director_prompt)
    feedback = " ".join(_words(subject_v31._extract_repair(director_prompt))[:40])
    negative = bounded_negative_prompt_v44(
        contract,
        reviewer_feedback=feedback,
    )
    grounded_v40.validate_grounded_visual_gate_v40(
        contract,
        compiled.compiled_prompt,
        negative,
    )
    return replace(
        compiled,
        negative_prompt=negative,
        compiler_version="visual-compiler-v44-priority-negative-budget",
    )


def scene_for_attempt_v44(
    scene: Any,
    *,
    scene_index: int,
    attempt: int,
    repair: str = "",
) -> Any:
    retried = convergence_v41.scene_for_attempt_v41(
        scene,
        scene_index=scene_index,
        attempt=attempt,
        repair=repair,
    )
    contract = convergence_v41.grounded_contract_for_v41(
        retried.image_prompt,
        scene_index,
    )
    negative = bounded_negative_prompt_v44(
        contract,
        reviewer_feedback=repair,
    )
    return replace(retried, negative_prompt=negative)


def validate_negative_budget_examples_v44() -> None:
    director = (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        "This makes it ideal for environments with strict data privacy requirements. "
        "Supporting source-grounded visual direction: A secure data center scene. "
        "Shot treatment: cause-to-result process view with visible directional change. "
        "V30 STORYBOARD: shot-4; validation"
    )
    compiled = compile_negative_budget_v44(director)
    if len(_words(compiled.negative_prompt)) > _MAX_NEGATIVE_WORDS:
        raise VisualNegativeBudgetError("Scene 4 negative prompt exceeded the v44 word budget")
    for phrase in ("readable text", "logo watermark", "collage split frame"):
        if phrase not in compiled.negative_prompt.casefold():
            raise VisualNegativeBudgetError(f"Mandatory negative phrase was dropped: {phrase}")


def install_production_visual_negative_budget_v44() -> None:
    """Install the final CLIP-safe negative compiler and failed-scene retry authority."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler

    validate_negative_budget_examples_v44()
    visual_prompt_compiler.compile_image_prompt = compile_negative_budget_v44
    image_generator.compile_image_prompt = compile_negative_budget_v44
    semantic_v28._scene_for_attempt = scene_for_attempt_v44
    _INSTALLED = True
