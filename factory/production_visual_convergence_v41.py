from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from . import production_visual_codex_gate_v36 as codex_v36
from . import production_visual_semantic_grounding_v40 as grounded_v40
from . import production_visual_semantic_review_v28 as semantic_v28
from . import production_visual_subject_authority_v31 as subject_v31
from .visual_storyboard_v30 import StoryboardFrame, clean, extract_claim, extract_scene_index


_INSTALLED = False
_MAX_WORDS = 58


class VisualConvergenceGateError(ValueError):
    """Raised before inference when the compact scene contract cannot remain literal."""


_CONCRETE_CONTROLLED_FRAME = StoryboardFrame(
    "controlled_test",
    2,
    "an indoor tabletop quality-control station with one small motor fixture and two plain plastic trays",
    "one lab technician at an indoor tabletop motor fixture",
    "the technician inserts one metal cylinder, presses one green button, and moves the cylinder into the left tray",
    "tight eye-level tabletop photograph showing the cylinder, green button, motor fixture, and both trays",
    "clean gray tabletop, black fixture, green button, white trays",
)

_CONCRETE_SOURCE_FRAME = StoryboardFrame(
    "source_grounded_ai",
    3,
    "an indoor office server room with a short rack and one black tabletop AI appliance",
    "one office IT technician beside a black tabletop AI appliance and short server rack",
    "the technician connects one blue Ethernet cable from the appliance to the rack and checks green status lights",
    "eye-level indoor photograph with the appliance, Ethernet cable, and rack filling the foreground",
    "black hardware, blue cable, green indicators, neutral office light",
)

_CONCRETE_BUSINESS_FRAME = StoryboardFrame(
    "business_adoption",
    3,
    "an indoor business server room with a short rack and one black tabletop local AI appliance",
    "one office IT technician beside a black tabletop local AI appliance and short server rack",
    "the technician connects one blue Ethernet cable from the appliance to the rack and checks green status lights",
    "eye-level indoor photograph with the appliance, Ethernet cable, and rack filling the foreground",
    "black hardware, blue cable, green indicators, neutral office light",
)

_FRAME_BANKS = dict(grounded_v40._FRAME_BANKS)
_source_frames = list(_FRAME_BANKS["source_grounded_ai"])
_source_frames[3] = _CONCRETE_SOURCE_FRAME
_FRAME_BANKS["source_grounded_ai"] = tuple(_source_frames)
_business_frames = list(_FRAME_BANKS["business_adoption"])
_business_frames[3] = _CONCRETE_BUSINESS_FRAME
_FRAME_BANKS["business_adoption"] = tuple(_business_frames)

_TOPIC_ANCHORS = dict(grounded_v40._TOPIC_ANCHORS)
_TOPIC_ANCHORS["controlled_test"] = "tabletop quality-control test on one metal part"
_TOPIC_ANCHORS["business_adoption"] = "enterprise local AI deployment on private business infrastructure"

_OBSERVED_NEGATIVES = {
    "source_grounded_ai": (
        "outdoor landscape",
        "mountain",
        "mountain peak",
        "valley",
        "lake",
        "bicycle",
        "motorcycle",
        "spacesuit",
        "astronaut",
        "science-fiction armor",
        "weapon",
        "gun",
        "rifle",
        "military uniform",
        "hiking scene",
        "futuristic corridor",
        "multi-panel layout",
        "contact sheet",
        "image mosaic",
    ),
    "business_adoption": (
        "outdoor landscape",
        "mountain",
        "bicycle",
        "spacesuit",
        "weapon",
        "robotics workshop",
        "public research hub",
        "factory floor",
        "multi-panel layout",
        "contact sheet",
        "image mosaic",
    ),
    "controlled_test": (
        "large industrial machinery",
        "factory floor",
        "open machine cabinet",
        "heavy equipment",
        "multiple workers",
        "robotic arm",
        "vehicle repair",
        "vintage laboratory",
        "black and white photograph",
    ),
}


def _words(value: str) -> list[str]:
    return codex_v36._words(value)


def _semantic_key(value: str) -> str:
    return " ".join(word.casefold() for word in _words(value))


def _contains_phrase(container: str, phrase: str) -> bool:
    container_key = _semantic_key(container)
    phrase_key = _semantic_key(phrase)
    return bool(phrase_key) and f" {phrase_key} " in f" {container_key} "


def classify_scene_v41(claim: str, direction: str = "") -> str:
    text = clean(f"{claim} {direction}").casefold()
    if any(
        term in text
        for term in (
            "before adoption",
            "controlled task",
            "test the claim",
            "repeatability",
            "failure rate",
            "controlled test",
        )
    ):
        return "controlled_test"
    if any(
        term in text
        for term in (
            "locally",
            "local deployment",
            "local ai",
            "own infrastructure",
            "on-premise",
            "on premise",
            "reduce reliance on cloud",
            "cloud services",
            "control over their data",
            "private infrastructure",
            "edge deployment",
            "customize their environments",
        )
    ):
        return "local_inference"
    if any(
        term in text
        for term in (
            "model release",
            "release of",
            "the release",
            "model's availability",
            "model availability",
            "new model",
            "flexibility and scalability",
            "enhance ai capabilities",
        )
    ):
        return "model_release"
    if any(
        term in text
        for term in (
            "companies",
            "businesses",
            "business workloads",
            "ai workloads",
            "adoption",
            "viable alternative",
            "organizations",
            "enterprise",
            "enterprise use cases",
            "custom solutions",
            "wide range of applications",
        )
    ):
        return "business_adoption"
    return "source_grounded_ai"


def _frame_for_v41(category: str, scene_index: int) -> StoryboardFrame:
    if category == "controlled_test":
        if scene_index % 3 == 2:
            return _CONCRETE_CONTROLLED_FRAME
        return codex_v36._CORRECTED_CONTROLLED_TEST_FRAMES[
            scene_index % len(codex_v36._CORRECTED_CONTROLLED_TEST_FRAMES)
        ]
    frames = _FRAME_BANKS[category]
    return frames[scene_index % len(frames)]


def grounded_contract_for_v41(
    director_prompt: str,
    scene_index: int | None = None,
) -> grounded_v40.GroundedSceneContractV40:
    claim = extract_claim(director_prompt)
    direction = semantic_v28._extract_direction(director_prompt)
    resolved_index = extract_scene_index(director_prompt, scene_index or 0)
    category = classify_scene_v41(claim, direction)
    frame = _frame_for_v41(category, resolved_index)
    frames = (
        codex_v36._CORRECTED_CONTROLLED_TEST_FRAMES
        if category == "controlled_test"
        else _FRAME_BANKS[category]
    )
    composition_index = (resolved_index // len(frames)) % len(grounded_v40._SHOT_COMPOSITIONS)
    composition_anchor = grounded_v40._SHOT_COMPOSITIONS[composition_index]
    topic_anchor = _TOPIC_ANCHORS[category]
    forbidden = (
        *grounded_v40._CATEGORY_NEGATIVES.get(category, ()),
        *_OBSERVED_NEGATIVES.get(category, ()),
    )
    if category == "controlled_test":
        forbidden = (
            *codex_v36.scene_contract_for_v36(frame).forbidden_substitutions,
            *forbidden,
        )
    return grounded_v40.GroundedSceneContractV40(
        identity=f"{category}-{frame.variant}-c{composition_index}-v41",
        category=category,
        variant=frame.variant,
        composition_anchor=composition_anchor,
        topic_anchor=topic_anchor,
        environment=frame.environment,
        subject=frame.subject,
        action=frame.action,
        camera=clean(f"{composition_anchor}; {frame.camera}"),
        palette=frame.palette,
        required_phrases=(topic_anchor, composition_anchor, frame.subject, frame.action),
        forbidden_substitutions=tuple(dict.fromkeys(forbidden)),
    )


def storyboard_frame_for_v41(
    director_prompt: str,
    scene_index: int | None = None,
) -> StoryboardFrame:
    contract = grounded_contract_for_v41(director_prompt, scene_index)
    return StoryboardFrame(
        contract.category,
        contract.variant,
        clean(f"{contract.topic_anchor}; {contract.environment}"),
        contract.subject,
        contract.action,
        contract.camera,
        contract.palette,
    )


def _feedback_negatives_v41(feedback: str) -> tuple[str, ...]:
    lowered = clean(feedback).casefold()
    mappings = (
        (("mountain", "peak", "valley", "landscape", "outdoor"), "outdoor mountain landscape"),
        (("spacesuit", "astronaut", "futuristic outfit", "sci-fi"), "spacesuit astronaut armor"),
        (("weapon", "gun", "rifle"), "weapon gun rifle"),
        (("bicycle", "bike", "motorcycle"), "bicycle motorcycle"),
        (("corridor",), "futuristic corridor"),
        (("factory", "industrial setting", "large machinery", "machine cabinet"), "factory floor heavy machinery"),
        (("two individuals", "multiple workers"), "multiple workers"),
        (("black and white", "monochrome"), "black and white vintage photograph"),
        (("collage", "split", "multiple panels", "multiple images"), "multi-panel contact sheet"),
    )
    inherited = codex_v36._feedback_negatives(feedback)
    observed = tuple(
        replacement
        for needles, replacement in mappings
        if any(needle in lowered for needle in needles)
    )
    return tuple(dict.fromkeys((*inherited, *observed)))


def scene_negative_prompt_v41(
    contract: grounded_v40.GroundedSceneContractV40,
    *,
    reviewer_feedback: str = "",
) -> str:
    candidates = (
        *grounded_v40._BASE_NEGATIVES,
        *contract.forbidden_substitutions,
        *_feedback_negatives_v41(reviewer_feedback),
    )
    kept: list[str] = []
    for candidate in candidates:
        normalized = _semantic_key(candidate)
        if not normalized or any(_semantic_key(item) == normalized for item in kept):
            continue
        if any(_contains_phrase(required, candidate) for required in contract.required_phrases):
            continue
        kept.append(clean(candidate))
    return ", ".join(kept)


def _pack_atomic(
    required_parts: Iterable[str],
    optional_parts: Iterable[str],
    *,
    limit: int,
) -> str:
    result: list[str] = []
    for part in required_parts:
        words = _words(part)
        if len(result) + len(words) > limit:
            raise VisualConvergenceGateError(
                f"Required convergent visual contract exceeds {limit} words: {clean(part)}"
            )
        result.extend(words)
    for part in optional_parts:
        words = _words(part)
        if len(result) + len(words) <= limit:
            result.extend(words)
    return " ".join(result).strip(" ,.;:") + "."


def compile_convergent_prompt_v41(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _MAX_WORDS,
) -> Any:
    """Keep literal indoor subjects and one-photograph geometry inside the CLIP budget."""
    del director_negative_prompt
    from .visual_prompt_compiler import CompiledVisualPrompt

    contract = grounded_contract_for_v41(director_prompt)
    feedback = " ".join(_words(subject_v31._extract_repair(director_prompt))[:40])
    budget = min(_MAX_WORDS, max(48, int(word_budget)))
    prompt = _pack_atomic(
        (
            "Photorealistic vertical documentary photograph",
            contract.topic_anchor,
            contract.composition_anchor,
            contract.subject,
            contract.action,
            "one uninterrupted photograph from one camera viewpoint",
        ),
        (
            contract.environment,
            contract.camera,
            "ordinary work clothes realistic hands mechanically plausible indoor hardware",
            contract.palette,
        ),
        limit=budget,
    )
    negative = scene_negative_prompt_v41(contract, reviewer_feedback=feedback)
    grounded_v40.validate_grounded_visual_gate_v40(contract, prompt, negative)
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=prompt,
        negative_prompt=negative,
        word_count=len(_words(prompt)),
        word_budget=budget,
        compiler_version="visual-compiler-v41-literal-convergence",
    )


def scene_for_attempt_v41(
    scene: Any,
    *,
    scene_index: int,
    attempt: int,
    repair: str = "",
) -> Any:
    base = semantic_v28._base_director_prompt(str(scene.image_prompt))
    base = clean(subject_v31._STORYBOARD_TAIL_RE.sub("", base))
    contract = grounded_contract_for_v41(base, scene_index)
    feedback = " ".join(_words(repair)[:40])
    suffix = f". V30 STORYBOARD: shot-{scene_index}; {contract.identity}"
    if feedback:
        suffix += f". V31 REPAIR: {feedback}"
    if attempt > 1:
        print(
            f"[visual-convergence-v41] retry scene {scene_index} attempt {attempt}: "
            f"{contract.identity}; reviewer={feedback or 'semantic mismatch'}",
            flush=True,
        )
    return replace(
        scene,
        image_prompt=base + suffix,
        negative_prompt=scene_negative_prompt_v41(
            contract,
            reviewer_feedback=feedback,
        ),
        seed=(int(scene.seed) + 86028121 * max(0, attempt - 1)) & 0x7FFFFFFF,
    )


def validate_convergence_examples_v41() -> None:
    scene_8 = (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        "Built on LiquidAI's technology, the model supports a wide range of applications, "
        "from custom solutions to enterprise use cases. Supporting source-grounded visual "
        "direction: A technical diagram of the model. Shot treatment: human-scale consequence. "
        "V30 STORYBOARD: shot-8; validation"
    )
    contract_8 = grounded_contract_for_v41(scene_8)
    if contract_8.category != "business_adoption":
        raise VisualConvergenceGateError("Enterprise use-case claim did not map to business adoption")
    compiled_8 = compile_convergent_prompt_v41(scene_8)
    if any(term in compiled_8.compiled_prompt.casefold() for term in ("deployment engineer", "rugged ai node")):
        raise VisualConvergenceGateError("Ambiguous outdoor-adventure vocabulary survived v41")

    scene_17 = (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        "Before adoption, read the linked source and test the claim on a controlled task. "
        "Supporting source-grounded visual direction: A user running an AI task locally. "
        "Shot treatment: human-scale consequence. V30 STORYBOARD: shot-17; validation"
    )
    compiled_17 = compile_convergent_prompt_v41(scene_17)
    for phrase in ("metal cylinder", "green button", "left tray"):
        if phrase not in compiled_17.compiled_prompt.casefold():
            raise VisualConvergenceGateError(f"Controlled-test action lost required phrase: {phrase}")


def install_production_visual_convergence_v41() -> None:
    """Install literal scene vocabulary, observed-defect negatives, and final retry authority."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler, visual_storyboard_v30
    from . import production_visual_atomic_storyboard_v34 as atomic_v34
    from . import production_visual_storyboard_priority_v30 as priority_v30
    from . import production_visual_storyboard_v30 as storyboard_v30

    validate_convergence_examples_v41()
    visual_storyboard_v30.storyboard_for = storyboard_frame_for_v41
    storyboard_v30.storyboard_for = storyboard_frame_for_v41
    subject_v31.storyboard_for = storyboard_frame_for_v41
    codex_v36.storyboard_for = storyboard_frame_for_v41
    grounded_v40.storyboard_frame_for_v40 = storyboard_frame_for_v41
    if hasattr(priority_v30, "storyboard_for"):
        priority_v30.storyboard_for = storyboard_frame_for_v41
    if hasattr(atomic_v34, "storyboard_for"):
        atomic_v34.storyboard_for = storyboard_frame_for_v41

    visual_prompt_compiler.compile_image_prompt = compile_convergent_prompt_v41
    image_generator.compile_image_prompt = compile_convergent_prompt_v41
    semantic_v28._scene_for_attempt = scene_for_attempt_v41
    _INSTALLED = True
