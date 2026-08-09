from __future__ import annotations

import re
from dataclasses import dataclass


_CLAIM_RE = re.compile(
    r"Factual technology documentary shot synchronized to this exact spoken sentence:\s*(.+?)\.\.?\s*Supporting source-grounded visual direction:",
    re.IGNORECASE,
)
_MARKER_RE = re.compile(r"V30 STORYBOARD:\s*shot-(\d+)", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StoryboardFrame:
    category: str
    variant: int
    environment: str
    subject: str
    action: str
    camera: str
    palette: str

    @property
    def identity(self) -> str:
        return f"{self.category}-{self.variant}"


_REGISTRY: dict[str, tuple[StoryboardFrame, ...]] = {
    "partnership_hub": (
        StoryboardFrame(
            "partnership_hub",
            0,
            "a contemporary public research hub with a visible compute aisle and an open robotics bay",
            "two diverse mid-career engineers beside a portable compute module and a compact industrial robot",
            "they connect the module into shared laboratory infrastructure while a third researcher verifies the physical link",
            "wide 28mm eye-level documentary view with the compute aisle on one side and robotics bay on the other",
            "graphite equipment, cool cyan indicators, warm practical light",
        ),
        StoryboardFrame(
            "partnership_hub",
            1,
            "a bright regional technology workshop with several clearly separated hands-on stations",
            "a small diverse team of adult instructors and researchers around one modular automation kit",
            "one instructor passes a sensor module to a researcher while others prepare distinct experiment stations",
            "medium-wide candid side view with layered foreground tools and visible workshop depth",
            "natural daylight, pale worktables, blue and amber equipment accents",
        ),
        StoryboardFrame(
            "partnership_hub",
            2,
            "an industrial equipment staging area inside a research facility",
            "three adult technical staff beside rugged compute cases, sensor kits, and a mobile robotics cart",
            "they unpack and route equipment toward separate regional research stations",
            "dynamic low-angle documentary view following the equipment flow across the floor",
            "neutral industrial daylight, charcoal cases, restrained orange safety accents",
        ),
    ),
    "compute_resources": (
        StoryboardFrame(
            "compute_resources",
            0,
            "a dark modern data-center aisle with blank black compute enclosures and fiber patch hardware",
            "two adult infrastructure engineers beside an open rack and a portable diagnostic sensor",
            "one connects fiber while the other verifies indicator lights on a blank compute enclosure",
            "long-lens aisle composition with strong depth and a clear foreground cable action",
            "deep graphite, cyan and green indicator lights, subtle warm skin tones",
        ),
        StoryboardFrame(
            "compute_resources",
            1,
            "a clean machine room containing large cooling pipes, blank compute cabinets, and cable trays",
            "one adult engineer and one research scientist beside a compact AI compute node",
            "they install the node into shared infrastructure and inspect its physical power and cooling connections",
            "wide environmental documentary view emphasizing scale, pipes, and the compact foreground module",
            "steel gray, cool white light, restrained blue highlights",
        ),
        StoryboardFrame(
            "compute_resources",
            2,
            "a high-performance computing integration bay with blank rack fronts and a robotic test fixture",
            "two adult engineers handling a GPU-sized compute module and a sensor harness",
            "they connect the compute module to the physical test fixture for an AI experiment",
            "tight diagonal process view from module to cable harness to robotic fixture",
            "dark navy equipment, cyan status lamps, warm amber task light",
        ),
    ),
    "access_knowledge": (
        StoryboardFrame(
            "access_knowledge",
            0,
            "a mobile training laboratory with a wheeled equipment cart and open floor space",
            "a mentor and two adult learners with sensor boards, small motors, and blank controller enclosures",
            "the mentor demonstrates how the same portable kit can be used at different learning stations",
            "medium eye-level view centered on the shared kit and natural hand interaction",
            "bright neutral daylight, blue cart, colorful but unlabelled components",
        ),
        StoryboardFrame(
            "access_knowledge",
            1,
            "a community technology classroom with modular worktables and physical AI experiment kits",
            "a diverse group of adult learners and one instructor",
            "learners assemble camera sensors and motor controllers while the instructor guides one concrete step",
            "wide classroom documentary view with one sharp foreground team and softer background teams",
            "warm daylight, light wood tables, blue and orange kit accents",
        ),
        StoryboardFrame(
            "access_knowledge",
            2,
            "a university maker space with shelves of blank component cases and a compact robotic arm",
            "one educator and two adult students around a practical automation prototype",
            "a student tests the robotic arm while the educator points to a physical sensor connection",
            "medium-close over-table view with the robotic action clearly visible",
            "clean white workspace, cool blue light, warm skin tones",
        ),
    ),
    "regional_network": (
        StoryboardFrame(
            "regional_network",
            0,
            "a large public research hall containing three distinct technology stations connected by overhead cable trays",
            "separate diverse adult teams at a compute rack, robotics station, and sensor test bench",
            "all three teams perform different tasks using the same shared infrastructure",
            "high-angle wide documentary view showing all stations in one continuous hall",
            "bright architectural daylight, blue infrastructure, varied equipment accents",
        ),
        StoryboardFrame(
            "regional_network",
            1,
            "a logistics bay for portable research infrastructure with rugged cases and mobile laboratory carts",
            "adult technical staff preparing identical compute kits for several destinations",
            "they load sensor cases and blank compute modules onto separate carts in an organized distribution flow",
            "diagonal floor-level composition with repeated carts creating geographic scale without a map",
            "neutral warehouse light, black cases, blue carts, amber safety accents",
        ),
        StoryboardFrame(
            "regional_network",
            2,
            "a fiber distribution room connected to adjacent research bays through visible cable routes",
            "two adult network engineers and researchers beside blank patch hardware",
            "they route multiple colored fiber bundles toward physically separated laboratory bays",
            "wide side view using the cable routes as leading lines across one coherent space",
            "graphite racks, vivid fiber colors, cool cyan practical lighting",
        ),
    ),
    "education": (
        StoryboardFrame(
            "education",
            0,
            "a modern engineering classroom with physical sensor kits and small wheeled robots",
            "an educator and a diverse group of adult students",
            "the educator demonstrates a repeatable robot navigation exercise while students prepare matching kits",
            "wide classroom view with the moving robot in the foreground and learners behind it",
            "natural daylight, pale tables, blue and yellow equipment accents",
        ),
        StoryboardFrame(
            "education",
            1,
            "a hands-on training bench inside a bright technical institute",
            "one adult learner and one mentor beside a programmable controller, camera sensor, and compact motor rig",
            "the learner wires the controller while the mentor checks the physical connection",
            "tight 50mm documentary view centered on natural hands and clearly separated components",
            "bright task light, neutral gray bench, blue component accents",
        ),
        StoryboardFrame(
            "education",
            2,
            "an open university laboratory with several different practical AI exercises",
            "small adult student teams and two instructors",
            "teams test a robotic gripper, an autonomous cart, and a sensor array at separate stations",
            "medium-high wide view showing distinct exercises without a collage layout",
            "clean white interior, balanced daylight, restrained colorful hardware",
        ),
    ),
    "expertise_support": (
        StoryboardFrame(
            "expertise_support",
            0,
            "a precision electronics integration bench with blank enclosures, probes, and a macro camera rig",
            "an experienced engineer mentoring a younger adult researcher",
            "they diagnose a sensor board and replace one physical connection together",
            "medium-close side view with the board, probes, and both faces in one coherent frame",
            "warm task lighting, dark bench, blue instrument accents",
        ),
        StoryboardFrame(
            "expertise_support",
            1,
            "a robotics maintenance bay with an industrial arm, tool cart, and calibration target",
            "two adult specialists and one research trainee",
            "a specialist demonstrates a calibration step while the trainee adjusts a camera sensor",
            "low three-quarter view emphasizing the industrial arm and human instruction",
            "cool industrial light, gray machinery, amber task lamp",
        ),
        StoryboardFrame(
            "expertise_support",
            2,
            "a clean data-storage hardware room with blank drive enclosures and fiber connections",
            "an adult data engineer and a research scientist",
            "they install a removable storage module and connect it to shared research infrastructure",
            "wide environmental view with the storage hardware foreground and research bay beyond",
            "graphite cabinets, green indicators, neutral white light",
        ),
    ),
    "controlled_test": (
        StoryboardFrame(
            "controlled_test",
            0,
            "a controlled robotics test cell with two identical unlabelled devices, sensors, and status lamps",
            "one adult evaluator beside the test fixture",
            "the evaluator runs the same physical task on both devices and observes the status lamps",
            "balanced comparison view in one continuous room with identical framing for both devices",
            "dark test cell, white fixtures, green and amber status lights",
        ),
        StoryboardFrame(
            "controlled_test",
            1,
            "a repeatability bench with a robotic gripper, camera sensor, and rows of identical test objects",
            "two adult researchers observing one automated cycle",
            "the gripper repeats the same pick-and-place action while sensors record the physical result",
            "tight diagonal view from test objects to gripper to observing researchers",
            "cool blue equipment, warm task light, neutral background",
        ),
        StoryboardFrame(
            "controlled_test",
            2,
            "a compact measurement station with a motor rig, optical sensor, and three physical result trays",
            "one adult engineer and one independent reviewer",
            "they compare repeated outcomes by sorting successful and failed parts into separate trays",
            "eye-level process view showing input, test action, and physical outcomes in one scene",
            "clean gray bench, blue rig, red and green result trays",
        ),
    ),
}


def clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip(" ,.;:")


def extract_claim(director_prompt: str) -> str:
    match = _CLAIM_RE.search(director_prompt)
    return clean(match.group(1)) if match else ""


def extract_scene_index(director_prompt: str, fallback: int = 0) -> int:
    match = _MARKER_RE.search(director_prompt)
    return int(match.group(1)) if match else int(fallback)


def classify_claim(claim: str) -> str:
    lowered = clean(claim).casefold()
    if any(term in lowered for term in ("before adoption", "controlled task", "test the claim", "repeatability", "failure rate")):
        return "controlled_test"
    if any(term in lowered for term in ("educator", "student", "education", "learning", "teach", "classroom")):
        return "education"
    if any(term in lowered for term in ("computing, data, software, and expertise", "expertise needed", "software and expertise")):
        return "expertise_support"
    if any(term in lowered for term in ("computing resources", "compute resources", "advanced computing", "data infrastructure")):
        return "compute_resources"
    if any(term in lowered for term in ("state and multistate", "nationwide", "across the us", "regional", "stronger foundation")):
        return "regional_network"
    if any(term in lowered for term in ("access", "tools and knowledge", "accessible", "knowledge")):
        return "access_knowledge"
    if any(term in lowered for term in ("joining", "partnership", "collaboration", "hubs program", "initiative")):
        return "partnership_hub"
    if any(term in lowered for term in ("expertise", "support", "research")):
        return "expertise_support"
    return "partnership_hub"


def storyboard_for(director_prompt: str, scene_index: int | None = None) -> StoryboardFrame:
    resolved_index = extract_scene_index(director_prompt, scene_index or 0)
    category = classify_claim(extract_claim(director_prompt))
    variants = _REGISTRY[category]
    return variants[resolved_index % len(variants)]


def storyboard_categories() -> tuple[str, ...]:
    return tuple(_REGISTRY)
