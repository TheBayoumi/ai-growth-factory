from __future__ import annotations

import math
from collections import Counter
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
    "an office equipment room with one compact AI appliance and short server rack",
    "one office IT technician beside the compact appliance",
    "the technician connects the appliance to the rack with one blue Ethernet cable",
    "eye-level indoor photograph with the appliance, Ethernet cable, and rack filling the foreground",
    "black hardware, blue cable, green indicators, neutral office light",
)

_CONCRETE_BUSINESS_FRAME = StoryboardFrame(
    "business_adoption",
    3,
    "an office equipment room with one compact local AI appliance and short server rack",
    "one office IT technician beside the compact appliance",
    "the technician connects the appliance to the rack with one blue Ethernet cable",
    "eye-level indoor photograph with the appliance, Ethernet cable, and rack filling the foreground",
    "black hardware, blue cable, green indicators, neutral office light",
)

_DIVERSE_MODEL_RELEASE_FRAMES = (
    StoryboardFrame(
        "model_release", 0,
        "a home-office desk with one laptop and one phone",
        "one developer beside the laptop and phone",
        "the developer runs the newly available model locally on both personal devices",
        "close eye-level view across the two devices",
        "natural daylight, graphite devices, soft blue accents",
    ),
    StoryboardFrame(
        "model_release", 1,
        "an electronics workbench with a compact computer and removable accelerator",
        "one systems engineer beside the open compact computer",
        "the engineer installs the accelerator and starts a local model validation",
        "tight diagonal process view across the workbench",
        "neutral task light, metallic hardware, cyan indicators",
    ),
    StoryboardFrame(
        "model_release", 2,
        "a robotics laboratory with a small mobile robot and edge computer",
        "one robotics researcher beside the robot and edge computer",
        "the researcher loads the model and starts an on-device navigation task",
        "wide laboratory view with the moving robot foreground",
        "bright laboratory light, white robot, blue equipment accents",
    ),
    StoryboardFrame(
        "model_release", 3,
        "a portable field kit with a rugged tablet and camera sensor",
        "one field engineer beside the open portable kit",
        "the engineer deploys the model to the tablet and tests the connected camera",
        "medium outdoor-shelter documentary view of the complete kit",
        "overcast daylight, rugged gray equipment, orange case accents",
    ),
)

_DIVERSE_LOCAL_FRAMES = (
    StoryboardFrame(
        "local_inference", 0,
        "a quiet home workspace with a laptop and phone disconnected from cloud hardware",
        "one developer beside the laptop and phone",
        "the developer completes an assistant task entirely on the personal devices",
        "close candid view with both devices visible",
        "warm window light, neutral desk, blue device accents",
    ),
    StoryboardFrame(
        "local_inference", 1,
        "a factory sensor station with one rugged edge computer",
        "one controls engineer beside the sensor line and edge computer",
        "the engineer processes live sensor input locally beside the machine",
        "diagonal view from sensor to edge computer to machine",
        "industrial gray, amber task light, cyan indicators",
    ),
    StoryboardFrame(
        "local_inference", 2,
        "a private clinic room with an offline medical workstation",
        "one clinician beside the workstation and unlabelled imaging device",
        "the clinician processes one image locally without sending it outside the room",
        "calm eye-level view of the private workflow",
        "clean white room, soft daylight, restrained teal accents",
    ),
    StoryboardFrame(
        "local_inference", 3,
        "a vehicle diagnostics bay with a compact edge computer",
        "one technician beside the vehicle and connected edge computer",
        "the technician analyzes sensor data locally while the vehicle remains connected",
        "low three-quarter view across vehicle, cable, and computer",
        "workshop daylight, dark equipment, blue cable accents",
    ),
    StoryboardFrame(
        "local_inference", 4,
        "a small retail back office with one local AI appliance",
        "one store technician beside the appliance and inventory camera",
        "the technician runs the camera workflow on the local appliance",
        "human-scale view with the shop floor softly visible beyond",
        "natural commercial light, black appliance, green indicators",
    ),
)

_AGENT_WORKFLOW_FRAMES = (
    StoryboardFrame(
        "agent_workflow", 0,
        "a robotics bench with a gripper, camera, and three physical tools",
        "one robotics developer beside the gripper and tools",
        "the developer starts a multi-step task as the gripper selects and uses each tool",
        "tight process view following the tool sequence",
        "bright task light, white bench, blue and orange tools",
    ),
    StoryboardFrame(
        "agent_workflow", 1,
        "a small smart-home test room with lights, blinds, and a local hub",
        "one test engineer beside the unbranded local hub",
        "the engineer gives one request and the hub coordinates three device actions",
        "wide room view showing the ordered physical changes",
        "warm interior light, pale walls, green hub indicator",
    ),
    StoryboardFrame(
        "agent_workflow", 2,
        "a research desk with a camera, microphone, robot, and laptop",
        "one researcher beside the connected tools",
        "the local agent moves from observation to tool choice to robot action",
        "over-table view with a clear left-to-right workflow",
        "neutral laboratory light, graphite devices, cyan accents",
    ),
    StoryboardFrame(
        "agent_workflow", 3,
        "a compact warehouse test lane with a mobile robot and sorting bins",
        "one automation engineer beside the robot and bins",
        "the robot inspects an object, chooses a bin, and completes the placement",
        "low tracking-style locked view down the test lane",
        "industrial daylight, gray robot, colored unlabelled bins",
    ),
    StoryboardFrame(
        "agent_workflow", 4,
        "a portable field-service shelter with a rugged tablet, camera sensor, and diagnostic meter",
        "one field technician beside the open service kit",
        "the technician lets the local agent inspect the camera feed, select the meter, and record the result",
        "medium documentary view with the complete tool sequence visible",
        "overcast daylight, rugged gray equipment, orange case accents",
    ),
    StoryboardFrame(
        "agent_workflow", 5,
        "a private clinic room with an offline workstation, scanner, and unlabelled imaging device",
        "one clinical technician beside the connected equipment",
        "the local agent checks one image, selects the scanner, and prepares the next offline step",
        "calm eye-level view across the private multi-device workflow",
        "clean white room, soft daylight, restrained teal accents",
    ),
    StoryboardFrame(
        "agent_workflow", 6,
        "a quiet home workspace with a laptop, phone, and unbranded USB instrument",
        "one developer beside the three personal devices",
        "the local agent reads one device result, chooses the instrument, and completes the task on the laptop",
        "close candid view with every device and action visible",
        "warm window light, graphite devices, blue instrument accents",
    ),
    StoryboardFrame(
        "agent_workflow", 7,
        "a benchmark bench with a laptop, compact desktop, and power meter",
        "one performance engineer beside the connected computers",
        "the local agent runs the same tool sequence on both computers while the meter records each result",
        "balanced comparison with both workflows equally visible",
        "neutral task light, graphite hardware, amber meter accents",
    ),
    StoryboardFrame(
        "agent_workflow", 8,
        "a manufacturing inspection line with a local vision computer, camera, and reject tray",
        "one quality engineer beside the inspection station",
        "the local agent inspects one part, selects the camera tool, and routes the part into the correct tray",
        "diagonal process view from camera to computer to tray",
        "industrial gray, white task light, cyan indicators",
    ),
    StoryboardFrame(
        "agent_workflow", 9,
        "an archival research table with unlabelled reports, a document scanner, and one laptop",
        "one analyst beside the ordered reports and connected scanner",
        "the local agent finds one report, selects the scanner tool, and adds the result to the laptop review",
        "high three-quarter view across the complete evidence workflow",
        "warm reading light, neutral paper, dark laptop",
    ),
)

_LONG_CONTEXT_FRAMES = (
    StoryboardFrame(
        "long_context", 0,
        "an archival research table with many unlabelled reports and one laptop",
        "one analyst beside the arranged reports and laptop",
        "the analyst connects evidence from early and late reports in one continuous review",
        "high three-quarter view across the full evidence spread",
        "warm reading light, neutral paper, dark laptop",
    ),
    StoryboardFrame(
        "long_context", 1,
        "an engineering project room with drawings, parts, and a workstation",
        "one systems engineer beside the project history and prototype",
        "the engineer traces a decision from old design material to the current prototype",
        "wide wall-to-bench documentary composition",
        "cool office light, pale drawings, metallic prototype",
    ),
    StoryboardFrame(
        "long_context", 2,
        "a legal review room with sealed case boxes and an offline workstation",
        "one reviewer beside the ordered case materials",
        "the reviewer compares distant parts of the case history without removing material",
        "balanced eye-level view of boxes, desk, and reviewer",
        "soft neutral light, brown boxes, graphite workstation",
    ),
    StoryboardFrame(
        "long_context", 3,
        "a laboratory notebook station beside a long-running physical experiment",
        "one scientist beside stacked records and the active experiment",
        "the scientist links earlier observations to the current sensor result",
        "diagonal view from records to experiment to sensor",
        "clean laboratory light, white records, blue sensors",
    ),
)

_PERFORMANCE_FRAMES = (
    StoryboardFrame(
        "performance", 0,
        "a benchmark bench with a laptop, compact desktop, and thermal sensors",
        "one performance engineer beside both computers",
        "the engineer runs the same model while sensors measure speed and heat",
        "balanced comparison with both computers equally visible",
        "neutral task light, graphite hardware, amber sensors",
    ),
    StoryboardFrame(
        "performance", 1,
        "a mobile-device lab with a phone, tablet, and power meter",
        "one device engineer beside the three connected instruments",
        "the engineer compares local generation while the power meter records each device",
        "tight over-table view centered on devices and meter",
        "bright laboratory light, black devices, cyan indicators",
    ),
    StoryboardFrame(
        "performance", 2,
        "a GPU test station with one accelerator server and cooling instrumentation",
        "one infrastructure engineer beside the server and airflow sensors",
        "the engineer increases concurrent workloads while monitoring the physical system",
        "wide technical view of server, cooling, and engineer",
        "dark hardware, green indicators, cool blue light",
    ),
    StoryboardFrame(
        "performance", 3,
        "a CPU evaluation desk with two compact computers and one stopwatch camera",
        "one evaluator beside the two computers",
        "the evaluator repeats the same local task on both machines for a fair comparison",
        "eye-level symmetric comparison in one room",
        "natural office light, silver and black computers, amber accent",
    ),
)

_DIVERSE_BUSINESS_FRAMES = (
    _CONCRETE_BUSINESS_FRAME,
    StoryboardFrame(
        "business_adoption", 1,
        "a manufacturing inspection line with a local vision computer",
        "one quality engineer beside the camera and local computer",
        "the engineer routes inspection images through the local system beside the line",
        "wide factory view with the inspected part foreground",
        "industrial gray, white task light, cyan indicators",
    ),
    StoryboardFrame(
        "business_adoption", 2,
        "a private clinic office with one local workstation and imaging device",
        "one clinical technician beside the workstation",
        "the technician evaluates one image locally inside the clinic",
        "calm eye-level documentary view of the private workflow",
        "soft daylight, clean white surfaces, teal accents",
    ),
    StoryboardFrame(
        "business_adoption", 3,
        "a logistics desk beside a warehouse lane and compact edge appliance",
        "one operations manager beside the appliance and parcel camera",
        "the manager starts a local sorting workflow as parcels move through the lane",
        "diagonal view connecting desk, appliance, and lane",
        "warehouse daylight, dark appliance, orange parcel accents",
    ),
)

_DIVERSE_SOURCE_FRAMES = (
    _CONCRETE_SOURCE_FRAME,
    _AGENT_WORKFLOW_FRAMES[2],
    _PERFORMANCE_FRAMES[0],
    _DIVERSE_LOCAL_FRAMES[1],
    _LONG_CONTEXT_FRAMES[0],
)

_FRAME_BANKS = {
    "local_inference": _DIVERSE_LOCAL_FRAMES,
    "model_release": _DIVERSE_MODEL_RELEASE_FRAMES,
    "business_adoption": _DIVERSE_BUSINESS_FRAMES,
    "source_grounded_ai": _DIVERSE_SOURCE_FRAMES,
    "agent_workflow": _AGENT_WORKFLOW_FRAMES,
    "long_context": _LONG_CONTEXT_FRAMES,
    "performance": _PERFORMANCE_FRAMES,
}

_TOPIC_ANCHORS = dict(grounded_v40._TOPIC_ANCHORS)
_TOPIC_ANCHORS["controlled_test"] = "tabletop quality-control test on one metal part"
_TOPIC_ANCHORS["business_adoption"] = "enterprise local AI deployment on private business infrastructure"
_TOPIC_ANCHORS["agent_workflow"] = "on-device agent completing a visible multi-step tool workflow"
_TOPIC_ANCHORS["long_context"] = "long-context model connecting distant evidence in one task"
_TOPIC_ANCHORS["performance"] = "measured local AI performance on everyday computing hardware"

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
            "tool calling", "tool use", "multi-step", "multi step", "agentic",
            "agent harness", "agents entirely", "agent workflow",
        )
    ):
        return "agent_workflow"
    if any(
        term in text
        for term in (
            "128k", "context window", "long context", "memory requirement",
            "training tokens", "distant evidence",
        )
    ):
        return "long_context"
    if any(
        term in text
        for term in (
            "tokens per second", "tok/s", "cpu", "gpu", "h100", "m5 max",
            "ryzen", "benchmark", "2.5 gb", "memory footprint", "inference speed",
        )
    ):
        return "performance"
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
            "released",
            "launched",
            "introduced",
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
    frame_index = resolved_index % len(frames)
    composition_index = (resolved_index // len(frames)) % len(grounded_v40._SHOT_COMPOSITIONS)
    camera_angles = (
        "eye-level three-quarter view",
        "restrained high-angle process view",
        "low shoulder-height documentary view",
    )
    camera_index = (
        resolved_index
        // (len(frames) * len(grounded_v40._SHOT_COMPOSITIONS))
    ) % len(camera_angles)
    composition_anchor = grounded_v40._SHOT_COMPOSITIONS[composition_index]
    camera_angle = camera_angles[camera_index]
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
        identity=f"{category}-{frame_index}-c{composition_index}-a{camera_index}-v41",
        category=category,
        variant=frame.variant,
        composition_anchor=composition_anchor,
        topic_anchor=topic_anchor,
        environment=frame.environment,
        subject=frame.subject,
        action=frame.action,
        camera=clean(f"{composition_anchor}; {camera_angle}; {frame.camera}"),
        palette=frame.palette,
        required_phrases=(topic_anchor, frame.environment, frame.subject, frame.action),
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


def _environment_family(environment: str) -> str:
    value = clean(environment).casefold()
    families = (
        ("performance-lab", ("benchmark", "gpu test", "cpu evaluation", "power meter")),
        ("business-operations", ("retail", "logistics", "warehouse", "business")),
        ("robotics", ("robot", "gripper", "smart-home", "smart home")),
        ("industrial-edge", ("factory", "manufacturing", "sensor line", "vehicle")),
        ("healthcare", ("clinic", "medical")),
        ("field-mobile", ("field kit", "portable", "rugged")),
        ("research-evidence", ("archival", "legal review", "project room", "notebook")),
        ("server-infrastructure", ("server room", "rack", "data center", "compute room")),
        ("controlled-test", ("quality-control", "test bench", "measurement bench")),
        ("personal-device", ("home", "laptop", "phone", "tablet", "personal device")),
    )
    for family, markers in families:
        if any(marker in value for marker in markers):
            return family
    return "general-workspace"


def validate_editorial_contract_diversity_v41(scenes: Iterable[Any]) -> dict[str, int]:
    """Reject a prompt set that collapses into one environment before any GPU inference."""
    scene_list = list(scenes)
    contracts = [
        grounded_contract_for_v41(str(scene.image_prompt), int(scene.scene_index))
        for scene in scene_list
    ]
    families = Counter(_environment_family(contract.environment) for contract in contracts)
    minimum_families = min(6, max(4, math.ceil(len(contracts) / 4)))
    maximum_family_count = max(4, math.ceil(len(contracts) * 0.35))
    if len(families) < minimum_families:
        raise VisualConvergenceGateError(
            f"Editorial plan uses only {len(families)} environment families; "
            f"requires at least {minimum_families}: {dict(families)}"
        )
    crowded = {name: count for name, count in families.items() if count > maximum_family_count}
    if crowded:
        raise VisualConvergenceGateError(
            "Editorial plan overuses one environment family: " + str(crowded)
        )
    identities = [contract.identity for contract in contracts]
    duplicates = [name for name, count in Counter(identities).items() if count > 1]
    if duplicates:
        raise VisualConvergenceGateError(
            "Editorial plan repeats executable shot contracts: " + ", ".join(duplicates)
        )
    return dict(families)


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
            contract.environment,
            contract.subject,
            contract.action,
            "one uninterrupted photograph",
        ),
        (
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
