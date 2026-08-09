from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from . import production_visual_codex_gate_v36 as codex_v36
from . import production_visual_semantic_review_v28 as semantic_v28
from . import production_visual_subject_authority_v31 as subject_v31
from .visual_storyboard_v30 import (
    StoryboardFrame,
    clean,
    extract_claim,
    extract_scene_index,
)


_INSTALLED = False
_MAX_WORDS = 58


class GroundedVisualGateError(ValueError):
    """Raised before GPU inference when a visual loses its narration-grounded contract."""


@dataclass(frozen=True)
class GroundedSceneContractV40:
    identity: str
    category: str
    variant: int
    composition_anchor: str
    topic_anchor: str
    environment: str
    subject: str
    action: str
    camera: str
    palette: str
    required_phrases: tuple[str, ...]
    forbidden_substitutions: tuple[str, ...]

    @property
    def review_setup(self) -> str:
        return clean(
            f"{self.topic_anchor}; {self.composition_anchor}; {self.subject}; {self.action}; {self.environment}; {self.camera}"
        )


_LOCAL_INFERENCE_FRAMES = (
    StoryboardFrame(
        "local_inference",
        0,
        "a private server room with one compact on-premise AI server and a local network switch",
        "one adult infrastructure engineer beside the compact AI server and private network switch",
        "the engineer connects the server to the local network and verifies active inference indicator lights",
        "eye-level medium-wide documentary view with the server and private network clearly visible",
        "graphite server hardware, restrained green indicators, neutral practical light",
    ),
    StoryboardFrame(
        "local_inference",
        1,
        "a secure office technology room with a GPU workstation and encrypted local storage enclosure",
        "one adult developer beside the GPU workstation and locked local storage enclosure",
        "the developer loads an AI model onto the workstation while the storage remains connected locally",
        "tight three-quarter documentary view centered on the workstation, storage, and cable connection",
        "dark neutral hardware, cool blue indicators, warm task light",
    ),
    StoryboardFrame(
        "local_inference",
        2,
        "a small private network cabinet containing two compact edge AI nodes and one unbranded router",
        "one adult IT administrator beside two edge AI nodes and the private router",
        "the administrator links both nodes through the router and checks local processing status lights",
        "wide eye-level view showing the complete private network path in one room",
        "charcoal enclosures, green network lights, clean white room light",
    ),
    StoryboardFrame(
        "local_inference",
        3,
        "an industrial edge-computing station beside a production sensor line",
        "one adult engineer beside a rugged edge AI computer connected to the sensor line",
        "the engineer attaches the sensor cable and starts local inference on the rugged computer",
        "diagonal process view from sensor cable to edge computer to active machinery",
        "rugged gray hardware, cyan indicators, restrained amber industrial light",
    ),
    StoryboardFrame(
        "local_inference",
        4,
        "a controlled data room with a local inference server and a locked storage array",
        "one adult data engineer beside the inference server and locked storage array",
        "the engineer routes data directly from the storage array into the local inference server",
        "medium-wide side view emphasizing the short private data path between both systems",
        "graphite cabinets, blue cable accents, soft neutral lighting",
    ),
    StoryboardFrame(
        "local_inference",
        5,
        "a compact small-business server closet with one local AI node and an office workstation",
        "one adult technician beside the local AI node and connected office workstation",
        "the technician connects the workstation directly to the local node without an external cloud link",
        "human-scale documentary view showing the workstation-to-server connection in one coherent room",
        "plain office hardware, green indicators, natural practical light",
    ),
    StoryboardFrame(
        "local_inference",
        6,
        "a private compute room containing one compact AI node and one larger rack server on the same local network",
        "one adult engineer between the compact AI node and larger rack server",
        "the engineer compares local inference on both machines through the same private network connection",
        "balanced comparison view with both compute systems large and equally visible",
        "dark navy hardware, cyan and green indicators, neutral room light",
    ),
)

_MODEL_RELEASE_FRAMES = (
    StoryboardFrame(
        "model_release",
        0,
        "a secure AI validation room with an unbranded local inference server and test workstation",
        "one adult machine-learning engineer beside the local server and test workstation",
        "the engineer installs a newly released AI model package and starts a controlled local validation run",
        "medium-close documentary view centered on the server connection and physical validation action",
        "graphite hardware, blue indicators, warm task light",
    ),
    StoryboardFrame(
        "model_release",
        1,
        "a compact compute integration bench with a GPU module and blank local server enclosure",
        "one adult systems engineer holding the GPU module beside the open local server",
        "the engineer installs the accelerator module to prepare the newly available model for local inference",
        "tight three-quarter process view with the module, server, and natural hands prominent",
        "dark server enclosure, metallic module, cool neutral lighting",
    ),
    StoryboardFrame(
        "model_release",
        2,
        "a private model-testing room containing two local compute nodes with different capacities",
        "one adult evaluator beside a compact compute node and a larger rack node",
        "the evaluator runs the same released model on both nodes and observes their physical status indicators",
        "balanced eye-level comparison within one continuous room",
        "charcoal nodes, green and amber indicators, soft white light",
    ),
    StoryboardFrame(
        "model_release",
        3,
        "an edge deployment bench with a rugged AI computer and removable encrypted storage module",
        "one adult deployment engineer beside the rugged computer and encrypted storage module",
        "the engineer transfers the released model from storage into the edge computer for offline deployment",
        "diagonal documentary view following the physical transfer path across the bench",
        "rugged gray hardware, blue storage accent, warm practical light",
    ),
    StoryboardFrame(
        "model_release",
        4,
        "a clean local AI operations room with three unbranded inference appliances",
        "two adult engineers beside three differently sized local inference appliances",
        "they configure the new model for flexible deployment across the three local machines",
        "wide environmental view with each appliance distinct and the engineers active in the foreground",
        "graphite appliances, restrained cyan indicators, neutral architectural light",
    ),
)

_BUSINESS_ADOPTION_FRAMES = (
    StoryboardFrame(
        "business_adoption",
        0,
        "a private company IT room with a compact local AI server beside existing business infrastructure",
        "one adult IT manager and one engineer beside the local AI server",
        "they inspect the private connection before approving local AI deployment for business workloads",
        "medium-wide candid documentary view with both people and the local server prominent",
        "neutral office hardware, green indicators, warm practical light",
    ),
    StoryboardFrame(
        "business_adoption",
        1,
        "a small operations room containing a local inference appliance and a standard cloud gateway appliance",
        "two adult technical decision-makers beside the two unbranded appliances",
        "they compare the local appliance with the external gateway before selecting the private deployment path",
        "balanced comparison view in one continuous environment",
        "dark gray appliances, blue and amber indicators, neutral light",
    ),
    StoryboardFrame(
        "business_adoption",
        2,
        "a secure business data room with a locked storage array and local AI compute node",
        "one adult data officer and one infrastructure engineer beside the storage and compute node",
        "they route a protected business dataset directly into the local AI node for a controlled workload",
        "diagonal process composition from locked storage to compute node",
        "graphite cabinets, blue cables, clean white light",
    ),
    StoryboardFrame(
        "business_adoption",
        3,
        "a compact retail back-office server room with one edge AI appliance",
        "one adult technician and one operations manager beside the edge AI appliance",
        "the technician activates a local workload while the manager observes the physical system status",
        "human-scale eye-level documentary view with the appliance large in the foreground",
        "plain commercial hardware, green status lamps, natural room light",
    ),
    StoryboardFrame(
        "business_adoption",
        4,
        "a manufacturing control room with one local AI server connected to production equipment",
        "one adult controls engineer and one plant manager beside the local server",
        "the engineer connects a production workload to the local server while the manager verifies the private setup",
        "wide contextual view showing server, people, and connected production equipment",
        "industrial gray equipment, cyan indicators, restrained amber light",
    ),
)

_SOURCE_GROUNDED_FRAMES = (
    StoryboardFrame(
        "source_grounded_ai",
        0,
        "a modern AI engineering workspace with one unbranded compute node and connected test workstation",
        "one adult AI engineer beside the compute node and workstation",
        "the engineer connects the hardware and begins a concrete model-processing task",
        "medium documentary view with one clear foreground action",
        "graphite hardware, blue indicators, warm practical light",
    ),
    StoryboardFrame(
        "source_grounded_ai",
        1,
        "a secure software integration room with a local server and modular storage appliance",
        "one adult systems engineer beside the server and storage appliance",
        "the engineer transfers a model package into the server and verifies the local hardware path",
        "tight three-quarter process view centered on the physical transfer",
        "dark neutral equipment, green indicators, clean white light",
    ),
    StoryboardFrame(
        "source_grounded_ai",
        2,
        "a compact compute laboratory containing two distinct unbranded AI appliances",
        "two adult engineers beside the two AI appliances",
        "they compare a controlled model task across both machines in one continuous workflow",
        "balanced eye-level comparison with both machines equally visible",
        "charcoal appliances, cyan and amber indicators, neutral light",
    ),
    StoryboardFrame(
        "source_grounded_ai",
        3,
        "an edge-computing workbench with a rugged AI node and connected sensor interface",
        "one adult deployment engineer beside the rugged AI node",
        "the engineer connects the interface and starts one local processing cycle",
        "diagonal documentary view from interface to AI node",
        "rugged gray hardware, blue cable accents, warm task light",
    ),
    StoryboardFrame(
        "source_grounded_ai",
        4,
        "a private operations room with a rack server and compact AI workstation",
        "one adult operator beside the rack server and AI workstation",
        "the operator routes a model workload from the workstation into the private server",
        "wide contextual view showing the full private workflow",
        "dark navy hardware, green indicators, soft neutral light",
    ),
)

_FRAME_BANKS = {
    "local_inference": _LOCAL_INFERENCE_FRAMES,
    "model_release": _MODEL_RELEASE_FRAMES,
    "business_adoption": _BUSINESS_ADOPTION_FRAMES,
    "source_grounded_ai": _SOURCE_GROUNDED_FRAMES,
}

_SHOT_COMPOSITIONS = (
    "tight foreground cable detail",
    "wide private infrastructure context",
    "diagonal hardware process view",
)

_TOPIC_ANCHORS = {
    "local_inference": "local AI inference on owned private infrastructure",
    "model_release": "new AI model release prepared for local deployment",
    "business_adoption": "business adoption of private local AI infrastructure",
    "source_grounded_ai": "source-grounded AI model processing on real compute hardware",
    "controlled_test": "controlled repeatability test with measurable physical outcomes",
}

_BASE_NEGATIVES = (
    "readable text",
    "pseudo-text",
    "gibberish",
    "logo",
    "watermark",
    "poster",
    "infographic",
    "chart",
    "collage",
    "split frame",
    "malformed anatomy",
    "extra limbs",
    "distorted hands",
    "warped equipment",
    "blurry image",
    "empty architecture",
)

_CATEGORY_NEGATIVES = {
    "local_inference": (
        "public research hub",
        "robotics workshop",
        "robotic arm",
        "training classroom",
        "electronics assembly bench",
        "warehouse equipment cart",
        "generic laboratory experiment",
    ),
    "model_release": (
        "public research hub",
        "robotics workshop",
        "classroom lesson",
        "warehouse distribution",
        "generic partnership meeting",
    ),
    "business_adoption": (
        "public research hub",
        "robotics classroom",
        "academic workshop",
        "warehouse equipment distribution",
        "generic laboratory experiment",
    ),
    "source_grounded_ai": (
        "public research hub",
        "generic partnership meeting",
        "classroom lesson",
        "warehouse equipment cart",
    ),
    "controlled_test": (),
}


def _words(value: str) -> list[str]:
    return codex_v36._words(value)


def _semantic_key(value: str) -> str:
    return " ".join(word.casefold() for word in _words(value))


def _contains_phrase(container: str, phrase: str) -> bool:
    container_key = _semantic_key(container)
    phrase_key = _semantic_key(phrase)
    return bool(phrase_key) and f" {phrase_key} " in f" {container_key} "


def classify_grounded_scene_v40(claim: str, direction: str = "") -> str:
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
        )
    ):
        return "business_adoption"
    return "source_grounded_ai"


def _controlled_frame(scene_index: int) -> StoryboardFrame:
    frames = codex_v36._CORRECTED_CONTROLLED_TEST_FRAMES
    return frames[scene_index % len(frames)]


def grounded_contract_for_v40(
    director_prompt: str,
    scene_index: int | None = None,
) -> GroundedSceneContractV40:
    claim = extract_claim(director_prompt)
    direction = semantic_v28._extract_direction(director_prompt)
    resolved_index = extract_scene_index(director_prompt, scene_index or 0)
    category = classify_grounded_scene_v40(claim, direction)
    if category == "controlled_test":
        frame = _controlled_frame(resolved_index)
        special = codex_v36.scene_contract_for_v36(frame)
        forbidden = special.forbidden_substitutions
    else:
        frames = _FRAME_BANKS[category]
        frame = frames[resolved_index % len(frames)]
        forbidden = _CATEGORY_NEGATIVES[category]
    topic_anchor = _TOPIC_ANCHORS[category]
    composition_index = (resolved_index // len(_FRAME_BANKS.get(category, (frame,)))) % len(_SHOT_COMPOSITIONS)
    composition_anchor = _SHOT_COMPOSITIONS[composition_index]
    return GroundedSceneContractV40(
        identity=f"{category}-{frame.variant}-c{composition_index}",
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
        forbidden_substitutions=tuple(forbidden),
    )


def storyboard_frame_for_v40(
    director_prompt: str,
    scene_index: int | None = None,
) -> StoryboardFrame:
    contract = grounded_contract_for_v40(director_prompt, scene_index)
    return StoryboardFrame(
        contract.category,
        contract.variant,
        clean(f"{contract.topic_anchor}; {contract.environment}"),
        contract.subject,
        contract.action,
        contract.camera,
        contract.palette,
    )


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
            raise GroundedVisualGateError(
                f"Required narration-grounded visual contract exceeds {limit} words: {clean(part)}"
            )
        result.extend(words)
    for part in optional_parts:
        words = _words(part)
        if len(result) + len(words) <= limit:
            result.extend(words)
    return " ".join(result).strip(" ,.;:") + "."


def _feedback_negatives(feedback: str) -> tuple[str, ...]:
    return codex_v36._feedback_negatives(feedback)


def scene_negative_prompt_v40(
    contract: GroundedSceneContractV40,
    *,
    reviewer_feedback: str = "",
) -> str:
    candidates = (
        *_BASE_NEGATIVES,
        *contract.forbidden_substitutions,
        *_feedback_negatives(reviewer_feedback),
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


def validate_grounded_visual_gate_v40(
    contract: GroundedSceneContractV40,
    compiled_prompt: str,
    negative_prompt: str,
) -> None:
    missing = [
        phrase
        for phrase in contract.required_phrases
        if not _contains_phrase(compiled_prompt, phrase)
    ]
    conflicts = [
        phrase
        for phrase in contract.required_phrases
        if _contains_phrase(negative_prompt, phrase)
    ]
    substitutions = [
        phrase
        for phrase in contract.forbidden_substitutions
        if _contains_phrase(compiled_prompt, phrase)
    ]
    if missing or conflicts or substitutions:
        raise GroundedVisualGateError(
            f"Narration-grounded scene {contract.identity} failed: missing={missing}; "
            f"negative_conflicts={conflicts}; forbidden_positive={substitutions}"
        )


def compile_grounded_prompt_v40(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _MAX_WORDS,
) -> Any:
    """Compile the exact narration topic, physical subject, and action before style clauses."""
    del director_negative_prompt
    from .visual_prompt_compiler import CompiledVisualPrompt

    contract = grounded_contract_for_v40(director_prompt)
    reviewer_feedback = " ".join(_words(subject_v31._extract_repair(director_prompt))[:24])
    positive_repair = (
        "make the required topic subject and physical action unmistakable"
        if reviewer_feedback
        else ""
    )
    treatment = semantic_v28._extract_treatment(director_prompt)
    camera = semantic_v28._camera_language(treatment) or contract.camera
    budget = min(_MAX_WORDS, max(48, int(word_budget)))
    prompt = _pack_atomic(
        (
            "Photorealistic vertical technology documentary photograph",
            contract.topic_anchor,
            contract.composition_anchor,
            contract.subject,
            contract.action,
        ),
        (
            positive_repair,
            contract.environment,
            camera,
            "one coherent full-frame scene with realistic anatomy and mechanically plausible hardware",
            contract.palette,
        ),
        limit=budget,
    )
    negative = scene_negative_prompt_v40(
        contract, reviewer_feedback=reviewer_feedback
    )
    validate_grounded_visual_gate_v40(contract, prompt, negative)
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=prompt,
        negative_prompt=negative,
        word_count=len(_words(prompt)),
        word_budget=budget,
        compiler_version="visual-compiler-v40-narration-grounded-codex-gate",
    )


def scene_for_attempt_v40(
    scene: Any,
    *,
    scene_index: int,
    attempt: int,
    repair: str = "",
) -> Any:
    """Regenerate a rejected scene from its immutable narration-grounded contract only."""
    base = semantic_v28._base_director_prompt(str(scene.image_prompt))
    base = clean(subject_v31._STORYBOARD_TAIL_RE.sub("", base))
    contract = grounded_contract_for_v40(base, scene_index)
    suffix = f". V30 STORYBOARD: shot-{scene_index}; {contract.identity}"
    feedback = " ".join(_words(repair)[:24])
    if feedback:
        suffix += f". V31 REPAIR: {feedback}"
    if attempt > 1:
        print(
            f"[grounded-visual-v40] retry scene {scene_index} attempt {attempt}: "
            f"{contract.identity}; reviewer={feedback or 'semantic mismatch'}",
            flush=True,
        )
    return replace(
        scene,
        image_prompt=base + suffix,
        negative_prompt=scene_negative_prompt_v40(
            contract,
            reviewer_feedback=feedback,
        ),
        seed=(int(scene.seed) + 49979687 * max(0, attempt - 1)) & 0x7FFFFFFF,
    )


def validate_grounded_examples_v40() -> None:
    examples = (
        (
            "Hugging Face has introduced a new approach to deploying AI models locally",
            "A screen showing a model deployment interface with local infrastructure highlighted",
            "local_inference",
        ),
        (
            "The release is part of a larger initiative to enhance AI capabilities",
            "A newly released model being prepared for deployment",
            "model_release",
        ),
        (
            "This change could impact how businesses handle AI workloads",
            "A company evaluating private AI infrastructure",
            "business_adoption",
        ),
    )
    for index, (claim, direction, expected) in enumerate(examples):
        director = (
            "Factual technology documentary shot synchronized to this exact spoken sentence: "
            f"{claim}. Supporting source-grounded visual direction: {direction}. "
            f"Shot treatment: documentary view. V30 STORYBOARD: shot-{index}; validation"
        )
        contract = grounded_contract_for_v40(director)
        if contract.category != expected:
            raise GroundedVisualGateError(
                f"Grounded visual classifier returned {contract.category}, expected {expected}"
            )
        compiled = compile_grounded_prompt_v40(director)
        validate_grounded_visual_gate_v40(
            contract,
            compiled.compiled_prompt,
            compiled.negative_prompt,
        )


def install_production_visual_semantic_grounding_v40() -> None:
    """Install narration-grounded storyboard, compiler, reviewer target, and retry authority."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler, visual_storyboard_v30
    from . import production_visual_atomic_storyboard_v34 as atomic_v34
    from . import production_visual_storyboard_priority_v30 as priority_v30
    from . import production_visual_storyboard_v30 as storyboard_v30

    validate_grounded_examples_v40()
    visual_storyboard_v30.storyboard_for = storyboard_frame_for_v40
    storyboard_v30.storyboard_for = storyboard_frame_for_v40
    subject_v31.storyboard_for = storyboard_frame_for_v40
    codex_v36.storyboard_for = storyboard_frame_for_v40
    if hasattr(priority_v30, "storyboard_for"):
        priority_v30.storyboard_for = storyboard_frame_for_v40
    if hasattr(atomic_v34, "storyboard_for"):
        atomic_v34.storyboard_for = storyboard_frame_for_v40

    visual_prompt_compiler.compile_image_prompt = compile_grounded_prompt_v40
    image_generator.compile_image_prompt = compile_grounded_prompt_v40
    semantic_v28._scene_for_attempt = scene_for_attempt_v40
    _INSTALLED = True
