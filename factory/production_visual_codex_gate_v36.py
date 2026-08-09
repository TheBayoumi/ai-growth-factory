from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from . import production_visual_semantic_review_v28 as semantic_v28
from . import production_visual_subject_authority_v31 as subject_v31
from .visual_storyboard_v30 import StoryboardFrame, clean, storyboard_for


_INSTALLED = False
_MAX_WORDS = 52
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")


class CodexVisualGateError(ValueError):
    """Raised before GPU inference when a compiled scene violates its immutable contract."""


@dataclass(frozen=True)
class SceneContractV36:
    identity: str
    environment: str
    subject: str
    action: str
    camera: str
    required_phrases: tuple[str, ...]
    forbidden_substitutions: tuple[str, ...]


_CORRECTED_CONTROLLED_TEST_FRAMES = (
    StoryboardFrame(
        "controlled_test",
        0,
        "a controlled industrial laboratory bench with visible cables and two identical unlabelled benchtop motor-test fixtures",
        "one adult evaluator beside two identical unlabelled benchtop motor-test fixtures",
        "the evaluator presses one physical start button and observes embedded green and amber indicator LEDs on both fixtures",
        "eye-level medium-wide single photograph with both fixtures equally visible",
        "dark neutral bench, white fixtures, restrained green and amber indicators",
    ),
    StoryboardFrame(
        "controlled_test",
        1,
        "a clean laboratory repeatability bench with a short aluminum rail",
        "one compact robotic gripper above four blank calibration cubes beneath one overhead optical sensor",
        "the robotic gripper moves one calibration cube from the first position to the second while the optical sensor observes",
        "tight three-quarter single photograph with the gripper, cubes, rail, and optical sensor filling the foreground",
        "cool blue machinery, warm task light, plain neutral background",
    ),
    StoryboardFrame(
        "controlled_test",
        2,
        "a compact industrial measurement bench with one motor rig and two plain result trays",
        "one adult engineer beside the motor rig and two unlabelled result trays",
        "the engineer places one tested part into a result tray after the motor cycle finishes",
        "eye-level single photograph showing the motor rig, tested part, and both trays",
        "clean gray bench, blue motor rig, restrained red and green tray accents",
    ),
)


_SPECIAL_CONTRACTS: dict[str, SceneContractV36] = {
    "controlled_test-0": SceneContractV36(
        identity="controlled_test-0",
        environment=_CORRECTED_CONTROLLED_TEST_FRAMES[0].environment,
        subject=_CORRECTED_CONTROLLED_TEST_FRAMES[0].subject,
        action=_CORRECTED_CONTROLLED_TEST_FRAMES[0].action,
        camera=_CORRECTED_CONTROLLED_TEST_FRAMES[0].camera,
        required_phrases=(
            "adult evaluator",
            "two identical unlabelled benchtop motor-test fixtures",
            "physical start button",
            "embedded green and amber indicator LEDs",
        ),
        forbidden_substitutions=(
            "laptop",
            "computer screen",
            "desk lamp",
            "hanging lamp",
            "product display",
            "catalog illustration",
            "diagram",
            "sketch",
        ),
    ),
    "controlled_test-1": SceneContractV36(
        identity="controlled_test-1",
        environment=_CORRECTED_CONTROLLED_TEST_FRAMES[1].environment,
        subject=_CORRECTED_CONTROLLED_TEST_FRAMES[1].subject,
        action=_CORRECTED_CONTROLLED_TEST_FRAMES[1].action,
        camera=_CORRECTED_CONTROLLED_TEST_FRAMES[1].camera,
        required_phrases=(
            "compact robotic gripper",
            "four blank calibration cubes",
            "short aluminum rail",
            "overhead optical sensor",
        ),
        forbidden_substitutions=(
            "DSLR",
            "photography camera",
            "tripod",
            "camera lens",
            "handheld device",
            "product photography",
            "display pedestal",
            "split view",
        ),
    ),
    "controlled_test-2": SceneContractV36(
        identity="controlled_test-2",
        environment=_CORRECTED_CONTROLLED_TEST_FRAMES[2].environment,
        subject=_CORRECTED_CONTROLLED_TEST_FRAMES[2].subject,
        action=_CORRECTED_CONTROLLED_TEST_FRAMES[2].action,
        camera=_CORRECTED_CONTROLLED_TEST_FRAMES[2].camera,
        required_phrases=(
            "adult engineer",
            "motor rig",
            "tested part",
            "two unlabelled result trays",
        ),
        forbidden_substitutions=(
            "laptop",
            "desk lamp",
            "product display",
            "diagram",
            "chart",
            "readable labels",
        ),
    ),
}

_BASE_NEGATIVES = (
    "readable text",
    "pseudo-text",
    "gibberish",
    "logo",
    "watermark",
    "collage",
    "split frame",
    "infographic",
    "diagram",
    "malformed anatomy",
    "extra limbs",
    "warped equipment",
    "blurry image",
)


def _words(value: str) -> list[str]:
    return _WORD_RE.findall(clean(value))


def _casefold(value: str) -> str:
    return clean(value).casefold()


def scene_contract_for_v36(frame: StoryboardFrame) -> SceneContractV36:
    special = _SPECIAL_CONTRACTS.get(frame.identity)
    if special is not None:
        return special
    return SceneContractV36(
        identity=frame.identity,
        environment=frame.environment,
        subject=frame.subject,
        action=frame.action,
        camera=frame.camera,
        required_phrases=(frame.subject, frame.action),
        forbidden_substitutions=(),
    )


def _feedback_negatives(feedback: str) -> tuple[str, ...]:
    lowered = _casefold(feedback)
    mappings = (
        (("camera", "tripod", "lens"), "photography camera"),
        (("laptop", "computer"), "laptop"),
        (("lamp",), "desk lamp"),
        (("product display", "product shot"), "product display"),
        (("diagram", "illustration", "sketch"), "catalog illustration"),
        (("collage", "split", "two separate images"), "split frame"),
        (("hand holding", "handheld"), "handheld device"),
    )
    return tuple(
        replacement
        for needles, replacement in mappings
        if any(needle in lowered for needle in needles)
    )


def scene_negative_prompt_v36(
    contract: SceneContractV36,
    *,
    reviewer_feedback: str = "",
) -> str:
    candidates = (
        *_BASE_NEGATIVES,
        *contract.forbidden_substitutions,
        *_feedback_negatives(reviewer_feedback),
    )
    required = tuple(_casefold(item) for item in contract.required_phrases)
    kept: list[str] = []
    for candidate in candidates:
        normalized = _casefold(candidate)
        if not normalized or normalized in {_casefold(item) for item in kept}:
            continue
        if any(normalized == phrase or normalized in phrase for phrase in required):
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
            raise CodexVisualGateError(
                f"Required visual contract exceeds the {limit}-word CLIP budget: {clean(part)}"
            )
        result.extend(words)
    for part in optional_parts:
        words = _words(part)
        if len(result) + len(words) <= limit:
            result.extend(words)
    return " ".join(result).strip(" ,.;:") + "."


def validate_codex_visual_gate_v36(
    contract: SceneContractV36,
    compiled_prompt: str,
    negative_prompt: str,
) -> None:
    positive = _casefold(compiled_prompt)
    negative = _casefold(negative_prompt)
    missing = [
        phrase
        for phrase in contract.required_phrases
        if _casefold(phrase) not in positive
    ]
    conflicts = [
        phrase
        for phrase in contract.required_phrases
        if _casefold(phrase) in negative
    ]
    substitutions = [
        phrase
        for phrase in contract.forbidden_substitutions
        if _casefold(phrase) in positive
    ]
    if missing or conflicts or substitutions:
        raise CodexVisualGateError(
            f"Scene contract {contract.identity} failed: missing={missing}; "
            f"negative_conflicts={conflicts}; forbidden_positive={substitutions}"
        )


def compile_codex_reviewed_prompt_v36(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _MAX_WORDS,
) -> Any:
    """Compile immutable subject/action clauses first and fail before GPU inference on drift."""
    del director_negative_prompt
    from .visual_prompt_compiler import CompiledVisualPrompt

    frame = storyboard_for(director_prompt)
    contract = scene_contract_for_v36(frame)
    reviewer_feedback = subject_v31._extract_repair(director_prompt)
    budget = min(_MAX_WORDS, max(44, int(word_budget)))
    prompt = _pack_atomic(
        (
            "Photorealistic vertical technology documentary photograph",
            contract.subject,
            contract.action,
        ),
        (
            contract.environment,
            contract.camera,
            "one uninterrupted scene with mechanically plausible geometry and natural laboratory lighting",
            "exact apparatus only" if reviewer_feedback else "",
        ),
        limit=budget,
    )
    negative = scene_negative_prompt_v36(
        contract,
        reviewer_feedback=reviewer_feedback,
    )
    validate_codex_visual_gate_v36(contract, prompt, negative)
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=prompt,
        negative_prompt=negative,
        word_count=len(_words(prompt)),
        word_budget=budget,
        compiler_version="visual-compiler-v36-codex-scene-contract-gate",
    )


def grounded_retry_instruction_v36(
    *,
    contract: SceneContractV36,
    reviewer_feedback: str,
) -> str:
    feedback = " ".join(_words(reviewer_feedback)[:24])
    return clean(
        f"V36 CONTRACT {contract.identity}; preserve exact subject and action; "
        f"reviewer defect: {feedback or 'semantic mismatch'}"
    )


def scene_for_attempt_v36(
    scene: Any,
    *,
    scene_index: int,
    attempt: int,
    repair: str = "",
) -> Any:
    """Regenerate only the failed scene from its immutable base and exact scene contract."""
    base = semantic_v28._base_director_prompt(str(scene.image_prompt))
    base = clean(subject_v31._STORYBOARD_TAIL_RE.sub("", base))
    frame = storyboard_for(base, scene_index)
    contract = scene_contract_for_v36(frame)
    feedback = clean(repair)
    suffix = f". V30 STORYBOARD: shot-{scene_index}; {contract.identity}"
    if feedback:
        suffix += (
            ". V31 REPAIR: "
            + grounded_retry_instruction_v36(
                contract=contract,
                reviewer_feedback=feedback,
            )
        )
    negative = scene_negative_prompt_v36(contract, reviewer_feedback=feedback)
    if attempt > 1:
        print(
            f"[codex-visual-gate] retry scene {scene_index} attempt {attempt}: {contract.identity}",
            flush=True,
        )
    return replace(
        scene,
        image_prompt=base + suffix,
        negative_prompt=negative,
        seed=(int(scene.seed) + 32452843 * max(0, attempt - 1)) & 0x7FFFFFFF,
    )


def validate_codex_controlled_test_registry_v36() -> None:
    for index, frame in enumerate(_CORRECTED_CONTROLLED_TEST_FRAMES):
        contract = scene_contract_for_v36(frame)
        director = (
            "Factual technology documentary shot synchronized to this exact spoken sentence: "
            "Before adoption test the claim on a controlled task. "
            "Supporting source-grounded visual direction: controlled physical evaluation. "
            f"Shot treatment: documentary view. V30 STORYBOARD: shot-{index}; {frame.identity}"
        )
        compiled = compile_codex_reviewed_prompt_v36(director)
        validate_codex_visual_gate_v36(
            contract,
            compiled.compiled_prompt,
            compiled.negative_prompt,
        )


def install_production_visual_codex_gate_v36() -> None:
    """Install the final scene-contract compiler and reviewer-driven regeneration authority."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler, visual_storyboard_v30

    visual_storyboard_v30._REGISTRY["controlled_test"] = _CORRECTED_CONTROLLED_TEST_FRAMES
    visual_prompt_compiler.compile_image_prompt = compile_codex_reviewed_prompt_v36
    image_generator.compile_image_prompt = compile_codex_reviewed_prompt_v36
    semantic_v28._scene_for_attempt = scene_for_attempt_v36
    validate_codex_controlled_test_registry_v36()
    _INSTALLED = True
